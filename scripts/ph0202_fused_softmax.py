# %%
import os

# Enable interpreter mode if no CUDA GPU is available
if not os.environ.get("TRITON_INTERPRET") and not __import__("torch").cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found — running in Triton interpreter mode")
print(f"Triton version: {triton.__version__}")
# %%
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # each program processes one complete row
    row_idx=tl.program_id(0)

    # Compute the pointer to the start of this program's row
    row_start_ptr=input_ptr+row_idx*input_row_stride

    # Generate column offsets: [0,1,2, ..., BLOCK_SIZE-1]
    col_offsets=tl.arange(0, BLOCK_SIZE)

    # Mask: only load the columns that actually exists
    mask=col_offsets<n_cols

    # Load the entire row 
    row=tl.load(row_start_ptr+ col_offsets, mask=mask, other=-float("inf"))

    # Step 1: Find the max value in the row (for numerical stability)
    row_max=tl.max(row, axis=0)

    # Step 2: Subtract the max (prevents overflow in exp)
    safe_row=row-row_max

    # step 3: Exponentiate
    numerator=tl.exp(safe_row)

    # step 4: Sum the exponential
    denominator=tl.sum(numerator, axis=0)

    # Normalize:
    softmax_output=numerator/denominator

    # Store the result
    output_start_ptr=output_ptr+row_idx*output_row_stride
    tl.store(output_start_ptr + col_offsets, softmax_output, mask=mask)

print("ciao")
# %%
