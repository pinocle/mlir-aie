//===- aie1_runtime_sequence.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --pass-pipeline='builtin.module(aie.device(aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu,aie-dma-to-npu))' %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: aie-opt --pass-pipeline='builtin.module(aie.device(aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu,aie-dma-to-npu))' %s -split-input-file -verify-diagnostics -o %t.mlir && aie-translate %t.mlir --aie-npu-to-binary --aie-output-binary=false --aie-device-name=seg --aie-sequence-name=copy | FileCheck %s --check-prefix=TXN

// An AIE1 (xcvc1902) runtime sequence. Shim BDs are five words
// (aie-rt NOC_DMABD*): address low, length in words, address high | valid,
// AXI (secure access, AxCACHE, burst), packet. The address patch names
// word 0. A push enables the channel and writes the BD id to its start queue
// (S2MM ch c at 0x1D140 + 8c, MM2S at 0x1D150 + 8c, queue at control + 4).
// AIE1 has no task-completion token: an await polls the direction's status
// register (0x1D160 / 0x1D164) until the channel's queue size, state and stall
// bits are clear.

// CHECK-DAG: memref.global "private" constant @[[IN:.*]] : memref<5xi32> = dense<[0, 64, 1, 18, 0]>
// CHECK-DAG: memref.global "private" constant @[[OUT:.*]] : memref<5xi32> = dense<[64, 32, 1, 18, 0]>
// CHECK-LABEL: aie.runtime_sequence @copy
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 16896000 : ui32} : memref<5xi32>
// CHECK: aiex.npu.address_patch(%{{.*}} : i32) {addr = 16896000 : ui32, arg_idx = 0 : i32}
// CHECK: %[[CTRL0:.*]] = arith.constant 16896336 : i32
// CHECK: aiex.npu.write32(%[[CTRL0]], %{{.*}})
// CHECK: %[[QUEUE0:.*]] = arith.constant 16896340 : i32
// CHECK: aiex.npu.write32(%[[QUEUE0]], %{{.*}})
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 25284608 : ui32} : memref<5xi32>
// CHECK: %[[PLUS:.*]] = arith.constant 64 : i32
// CHECK: aiex.npu.address_patch(%[[PLUS]] : i32) {addr = 25284608 : ui32, arg_idx = 1 : i32}
// CHECK: %[[CTRL1:.*]] = arith.constant 25284936 : i32
// CHECK: aiex.npu.write32(%[[CTRL1]], %{{.*}})
// CHECK: %[[QUEUE1:.*]] = arith.constant 25284940 : i32
// CHECK: aiex.npu.write32(%[[QUEUE1]], %{{.*}})
// CHECK: %[[IDLE:.*]] = arith.constant 3628 : i32
// CHECK: %[[STATUS:.*]] = arith.constant 25284960 : i32
// CHECK: aiex.npu.maskpoll(%[[STATUS]], %{{.*}}, %[[IDLE]])
// CHECK-NOT: aiex.npu.sync

// TXN header: version 0.1, XAIE_DEV_GEN_AIE (1), 9 rows; 50 columns.
// TXN: 09010100
// TXN-NEXT: 00000032

module {
  aie.device(xcvc1902) @seg {
    %shim2 = aie.tile(2, 0)
    %shim3 = aie.tile(3, 0)
    aie.runtime_sequence @copy(%in: memref<64xi32>, %out: memref<64xi32>) {
      %t0 = aiex.dma_configure_task(%shim2, MM2S, 0) {
        aie.dma_bd(%in : memref<64xi32> offset = 0 len = 64) {burst_length = 64 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task(%shim3, S2MM, 1) {
        aie.dma_bd(%out : memref<64xi32> offset = 16 len = 32) {burst_length = 64 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }
}

// -----

// AIE1 shim BDs have no strides.
module {
  aie.device(xcvc1902) @seg {
    %shim2 = aie.tile(2, 0)
    aie.runtime_sequence @copy(%in: memref<128xi32>) {
      %t0 = aiex.dma_configure_task(%shim2, MM2S, 0) {
        // expected-error@+2 {{'aiex.npu.writebd' op sets a buffer descriptor field this tile's BD layout lacks}}
        // expected-error@+1 {{failed to legalize operation 'aiex.npu.writebd'}}
        aie.dma_bd(%in : memref<128xi32> offset = 0 len = 32 sizes = [4, 8] strides = [16, 1])
        aie.end
      }
      aiex.dma_start_task(%t0)
    }
  }
}
