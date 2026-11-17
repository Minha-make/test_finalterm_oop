"""Lớp cơ sở trừu tượng Layer cho toàn bộ các tầng trong mạng nơ-ron."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nnlib.tensor import Parameter, Tensor


class Layer(ABC):
    """Lớp cơ sở trừu tượng cho mọi tầng nơ-ron trong hệ thống.

    Vai trò trong Design Pattern: Component (Composite Pattern).
    Tất cả các tầng đơn lẻ (Leaf) như Dense, ReLU, Sigmoid, Dropout và chuỗi
    nhiều tầng (Composite) như Sequential đều kế thừa từ lớp này, cho phép
    Trainer và các thành phần khác tương tác đa hình không phân biệt tầng đơn
    hay khối kết hợp.
    """

    def __init__(self, name: str = "") -> None:
        """Khởi tạo tầng cơ sở.

        Args:
            name: Tên của tầng (nếu để trống sẽ mặc định dùng tên lớp).
        """
        self._name: str = name if name else self.__class__.__name__
        self._training: bool = True

    @property
    def name(self) -> str:
        """Tên định danh của tầng."""
        return self._name

    @property
    def training(self) -> bool:
        """Trạng thái hiện tại: True là chế độ huấn luyện, False là đánh giá."""
        return self._training

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Thực hiện lan truyền thuận (forward propagation).

        Args:
            x: Tensor đầu vào có kích thước theo batch-first (N, in_features).

        Returns:
            Tensor đầu ra sau khi tính toán.
        """
        ...

    @abstractmethod
    def backward(self, grad_out: Tensor) -> Tensor:
        """Thực hiện lan truyền ngược (backward propagation).

        Args:
            grad_out: Gradient dL/dy truyền ngược từ tầng kế tiếp về.

        Returns:
            Gradient dL/dx truyền tiếp về tầng trước.
        """
        ...

    def parameters(self) -> list[Parameter]:
        """Trả về danh sách tất cả các trọng số có thể học (Parameter) của tầng.

        Mặc định trả về danh sách rỗng (đối với các tầng không có tham số
        như Activation, Dropout). Các tầng có tham số (như Dense) hoặc tầng
        hợp thành (như Sequential) sẽ ghi đè phương thức này.

        Returns:
            Danh sách các đối tượng Parameter.
        """
        return []

    def train(self) -> None:
        """Chuyển tầng sang chế độ huấn luyện (Dropout ngắt nơ-ron)."""
        self._training = True

    def eval(self) -> None:
        """Chuyển sang chế độ đánh giá (Dropout giữ nguyên đầu vào)."""
        self._training = False

    def state_dict(self) -> dict[str, np.ndarray]:
        """Trích xuất từ điển trạng thái chứa các mảng trọng số để lưu trữ.

        Returns:
            Dictionary ánh xạ tên tham số sang mảng numpy ndarray.
        """
        params_dict: dict[str, np.ndarray] = {}
        for param in self.parameters():
            key = f"{self._name}.{param.name}" if param.name else self._name
            params_dict[key] = param.data
        return params_dict

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Nạp các mảng trọng số từ từ điển trạng thái vào các tham số của tầng.

        Args:
            state: Dictionary ánh xạ tên tham số sang mảng numpy ndarray.
        """
        for param in self.parameters():
            key = f"{self._name}.{param.name}" if param.name else self._name
            if key in state:
                param.data = state[key]
