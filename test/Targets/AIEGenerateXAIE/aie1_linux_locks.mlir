// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s
module {
  aie.device(xcvc1902) {
    %t = aie.tile(7, 2)
    %lock = aie.lock(%t, 0) {init = 1 : i32}
  }
}
// CHECK: ctx->XAieConfig = new XAie_Config{};
// CHECK: int mlir_aie_initialize_locks
// CHECK-NOT: XAie_LockSetValue
// CHECK: XAie_LockRelease(ctx->XAieDevInst, XAie_TileLoc(7,2), XAie_LockInit(0, 1), 0)
