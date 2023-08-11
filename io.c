/*
 * MLP DSP functions x86-optimized
 * Copyright (c) 2009 Ramiro Polla
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "io.h"

#include <stdint.h>

typedef int64_t x86_reg;

#ifdef __GNUC__
#    define AV_GCC_VERSION_AT_LEAST(x,y) (__GNUC__ > (x) || __GNUC__ == (x) && __GNUC_MINOR__ >= (y))
#    define AV_GCC_VERSION_AT_MOST(x,y)  (__GNUC__ < (x) || __GNUC__ == (x) && __GNUC_MINOR__ <= (y))
#else
#    define AV_GCC_VERSION_AT_LEAST(x,y) 0
#    define AV_GCC_VERSION_AT_MOST(x,y)  0
#endif

#if AV_GCC_VERSION_AT_LEAST(4,3) || defined(__clang__)
#    define av_cold __attribute__((cold))
#else
#    define av_cold
#endif

#define EXTERN_PREFIX ""
#define LABEL_MANGLE(a) EXTERN_PREFIX #a

#define AV_STRINGIFY(s)         AV_TOSTRING(s)
#define AV_TOSTRING(s) #s

#define MAX_CHANNELS                8

/** which multiple of 48000 the maximum sample rate is */
#define MAX_RATEFACTOR      4
/** maximum sample frequency seen in files */
#define MAX_SAMPLERATE      (MAX_RATEFACTOR * 48000)

/** maximum number of audio samples within one access unit */
#define MAX_BLOCKSIZE       (40 * MAX_RATEFACTOR)
/** next power of two greater than MAX_BLOCKSIZE */
#define MAX_BLOCKSIZE_POW2  (64 * MAX_RATEFACTOR)

/** number of allowed filters */
#define NUM_FILTERS         2

/** The maximum number of taps in IIR and FIR filters. */
#define MAX_FIR_ORDER       8
#define MAX_IIR_ORDER       4

typedef struct MLPDSPContext {
    void (*mlp_filter_channel)(int32_t *state, const int32_t *coeff,
                               int firorder, int iirorder,
                               unsigned int filter_shift, int32_t mask,
                               int blocksize, int32_t *sample_buffer);
    void (*mlp_rematrix_channel)(int32_t *samples,
                                 const int32_t *coeffs,
                                 const uint8_t *bypassed_lsbs,
                                 const int8_t *noise_buffer,
                                 int index,
                                 unsigned int dest_ch,
                                 uint16_t blockpos,
                                 unsigned int maxchan,
                                 int matrix_noise_shift,
                                 int access_unit_size_pow2,
                                 int32_t mask);
    int32_t (*(*mlp_select_pack_output)(uint8_t *ch_assign,
                                        int8_t *output_shift,
                                        uint8_t max_matrix_channel,
                                        int is32))(int32_t, uint16_t, int32_t (*)[], void *, uint8_t*, int8_t *, uint8_t, int);
    int32_t (*mlp_pack_output)(int32_t lossless_check_data,
                               uint16_t blockpos,
                               int32_t (*sample_buffer)[MAX_CHANNELS],
                               void *data,
                               uint8_t *ch_assign,
                               int8_t *output_shift,
                               uint8_t max_matrix_channel,
                               int is32);
} MLPDSPContext;

#define REMATRIX_CHANNEL_FUNC(opt) \
void ff_mlp_rematrix_channel_##opt(int32_t *samples, \
                                   const int32_t *coeffs, \
                                   const uint8_t *bypassed_lsbs, \
                                   const int8_t *noise_buffer, \
                                   int index, \
                                   unsigned int dest_ch, \
                                   uint16_t blockpos, \
                                   unsigned int maxchan, \
                                   int matrix_noise_shift, \
                                   int access_unit_size_pow2, \
                                   int32_t mask);

REMATRIX_CHANNEL_FUNC(sse4)
REMATRIX_CHANNEL_FUNC(avx2_bmi2)

extern char ff_mlp_firorder_8;
extern char ff_mlp_firorder_7;
extern char ff_mlp_firorder_6;
extern char ff_mlp_firorder_5;
extern char ff_mlp_firorder_4;
extern char ff_mlp_firorder_3;
extern char ff_mlp_firorder_2;
extern char ff_mlp_firorder_1;
extern char ff_mlp_firorder_0;

extern char ff_mlp_iirorder_4;
extern char ff_mlp_iirorder_3;
extern char ff_mlp_iirorder_2;
extern char ff_mlp_iirorder_1;
extern char ff_mlp_iirorder_0;

static const void * const firtable[9] = { &ff_mlp_firorder_0, &ff_mlp_firorder_1,
                                          &ff_mlp_firorder_2, &ff_mlp_firorder_3,
                                          &ff_mlp_firorder_4, &ff_mlp_firorder_5,
                                          &ff_mlp_firorder_6, &ff_mlp_firorder_7,
                                          &ff_mlp_firorder_8 };
static const void * const iirtable[5] = { &ff_mlp_iirorder_0, &ff_mlp_iirorder_1,
                                          &ff_mlp_iirorder_2, &ff_mlp_iirorder_3,
                                          &ff_mlp_iirorder_4 };

#define MLPMUL(label, offset, offs, offc)   \
    LABEL_MANGLE(label)":             \n\t" \
    "movslq "offset"+"offs"(%0), %%rax\n\t" \
    "movslq "offset"+"offc"(%1), %%rdx\n\t" \
    "imul                 %%rdx, %%rax\n\t" \
    "add                  %%rax, %%rsi\n\t"

#define FIRMULREG(label, offset, firc)\
    LABEL_MANGLE(label)":       \n\t" \
    "movslq "#offset"(%0), %%rax\n\t" \
    "imul        %"#firc", %%rax\n\t" \
    "add            %%rax, %%rsi\n\t"

#define CLEAR_ACCUM                   \
    "xor            %%rsi, %%rsi\n\t"

#define SHIFT_ACCUM                   \
    "shr     %%cl,         %%rsi\n\t"

#define ACCUM    "%%rdx"
#define RESULT   "%%rsi"
#define RESULT32 "%%esi"

#define BINC  AV_STRINGIFY(4* MAX_CHANNELS)
#define IOFFS AV_STRINGIFY(4*(MAX_FIR_ORDER + MAX_BLOCKSIZE))
#define IOFFC AV_STRINGIFY(4* MAX_FIR_ORDER)

#define FIRMUL(label, offset) MLPMUL(label, #offset,   "0",   "0")
#define IIRMUL(label, offset) MLPMUL(label, #offset, IOFFS, IOFFC)

static void mlp_filter_channel_x86(int32_t *state, const int32_t *coeff,
                                   int firorder, int iirorder,
                                   unsigned int filter_shift, int32_t mask,
                                   int blocksize, int32_t *sample_buffer)
{
    const void *firjump = firtable[firorder];
    const void *iirjump = iirtable[iirorder];

    blocksize = -blocksize;

    __asm__ volatile(
        "1:                           \n\t"
        CLEAR_ACCUM
        "jmp  *%5                     \n\t"
        FIRMUL   (ff_mlp_firorder_8, 0x1c   )
        FIRMUL   (ff_mlp_firorder_7, 0x18   )
        FIRMUL   (ff_mlp_firorder_6, 0x14   )
        FIRMUL   (ff_mlp_firorder_5, 0x10   )
        FIRMUL   (ff_mlp_firorder_4, 0x0c   )
        FIRMUL   (ff_mlp_firorder_3, 0x08   )
        FIRMUL   (ff_mlp_firorder_2, 0x04   )
        FIRMULREG(ff_mlp_firorder_1, 0x00, 8)
        LABEL_MANGLE(ff_mlp_firorder_0)":\n\t"
        "jmp  *%6                     \n\t"
        IIRMUL   (ff_mlp_iirorder_4, 0x0c   )
        IIRMUL   (ff_mlp_iirorder_3, 0x08   )
        IIRMUL   (ff_mlp_iirorder_2, 0x04   )
        IIRMUL   (ff_mlp_iirorder_1, 0x00   )
        LABEL_MANGLE(ff_mlp_iirorder_0)":\n\t"
        SHIFT_ACCUM
        "mov  "RESULT"  ,"ACCUM"      \n\t"
        "add  (%2)      ,"RESULT"     \n\t"
        "and   %4       ,"RESULT"     \n\t"
        "sub   $4       ,  %0         \n\t"
        "mov  "RESULT32", (%0)        \n\t"
        "mov  "RESULT32", (%2)        \n\t"
        "add $"BINC"    ,  %2         \n\t"
        "sub  "ACCUM"   ,"RESULT"     \n\t"
        "mov  "RESULT32","IOFFS"(%0)  \n\t"
        "incl              %3         \n\t"
        "js 1b                        \n\t"
        : /* 0*/"+r"(state),
          /* 1*/"+r"(coeff),
          /* 2*/"+r"(sample_buffer),
          /* 3*/"+r"(blocksize)
        : /* 4*/"r"((x86_reg)mask), /* 5*/"r"(firjump),
          /* 6*/"r"(iirjump)      , /* 7*/"c"(filter_shift)
        , /* 8*/"r"((int64_t)coeff[0]), "mp"(ff_mlp_firorder_8)
        : "rax", "rdx", "rsi"
    );
}

void ff_mlpdsp_init_x86(MLPDSPContext *c)
{
    c->mlp_filter_channel = mlp_filter_channel_x86;
}
