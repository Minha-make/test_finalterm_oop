"""Lớp FakeLayer giả lập một tầng nơ-ron phục vụ kiểm thử độc lập cho Phân hệ B."""

from __future__ import annotations

import numpy as np

from nnlib.exceptions import StateError
from nnlib.layers.base import Layer
from nnlib.tensor import Parameter, Tensor


class FakeLayer(Layer):
    """Tầng giả lập thực hiện phép nhân với trọng số vô hướng hoặc ma trận.

    Vai trò: Phục vụ kiểm thử độc lập cho Phân hệ B (SGD, Adam, Trainer)
    trong giai đoạn tuần 1-2 mà không cần phụ thuộc vào mã nguồn thật của Phân hệ A.
    """

    def __init__(self, init_weight: float = 2.0, name: str = "fake_layer") -> None:
        """Khởi tạo FakeLayer với trọng số vô hướng ban đầu.

        Args:
            init_weight: Giá trị khởi tạo của trọng số học được.
            name: Tên của tầng.
        """
        super().__init__(name=name)
        self._weight: Parameter = Parameter(
            data=np.array([init_weight], dtype=np.float32),
            name="scale",
        )
        self._cache_x: Tensor | None = None

    @property
    def weight(self) -> Parameter:
        """Tham số trọng số của tầng giả lập."""
        return self._weight

    def forward(self, x: Tensor) -> Tensor:
        """Lan truyền thuận: y = x * weight.

        Args:
            x: Tensor đầu vào.

        Returns:
            Tensor kết quả sau phép nhân.
        """
        self._cache_x = x
        out_data = x.data * self._weight.data
        return Tensor(out_data)

    def backward(self, grad_out: Tensor) -> Tensor:
        """Lan truyền ngược: dL/dx = grad_out * weight, dL/dw = sum(grad_out * x).

        Args:
            grad_out: Gradient truyền về từ phía sau.

        Returns:
            Gradient đối với đầu vào x.

        Raises:
            StateError: Nếu gọi backward trước khi gọi forward (Rule R2).
        """
        # Rule R2: backward chỉ được gọi sau khi đã forward có cache
        if self._cache_x is None:
            raise StateError("Không thể gọi backward khi chưa forward (cache rỗng).")

        grad_w = np.sum(grad_out.data * self._cache_x.data)
        if self._weight.grad is None:
            self._weight.grad = np.array([grad_w], dtype=np.float32)
        else:
            self._weight.grad = self._weight.grad + np.array([grad_w], dtype=np.float32)

        grad_x = grad_out.data * self._weight.data
        return Tensor(grad_x)

    def parameters(self) -> list[Parameter]:
        """Trả về danh sách chứa tham số trọng số duy nhất của tầng."""
        return [self._weight]
