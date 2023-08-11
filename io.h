#ifndef FOO_H
#define FOO_H

#ifdef __cplusplus
extern "C" {
#endif

struct MLPDSPContext;

typedef struct MLPDSPContext MLPDSPContext;

__attribute__((visibility("default"))) void ff_mlpdsp_init_x86(MLPDSPContext *c);

#ifdef __cplusplus
}
#endif

#endif
