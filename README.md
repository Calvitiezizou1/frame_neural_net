
**Deep Learning framework built from scratch in C++.**

Deconstruct the "black box" of Deep Learning.

This project is currently under active development.

### Implemented
*   **Back propagation**
    * **Autograd Engine:** Implementing the Chain Rule for backward propagation.
*   **Tensor Core:**
    *   Dynamic memory allocation for N-dimensional tensors.
    *   Shape management and stride calculation.
*   **Basic Arithmetic Operations:**
    *   Element-wise Addition, Subtraction, Multiplication.
    *   Scalar operations / Broadcasting basics.
*   **Linear Algebra:**
    *   **Matrix Multiplication (MatMul):** The core engine of neural networks.

### In Progress
*   **Flatten Module:** Implementing efficient tensor reshaping (transforming $(N, C, H, W)$ tensors into $(N, Features)$) to feed into Fully Connected layers.
