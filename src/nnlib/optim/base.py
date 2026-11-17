"""Giao diện trừu tượng IOptimizer cho các thuật toán tối ưu hóa."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from nnlib.exceptions import InvalidConfigError

if TYPE_CHECKING:
    from nnlib.tensor import Parameter


class IOptimizer(ABC):
    """Giao diện trừu tượng cho thuật toán tối ưu hóa trọng số.

    Vai trò trong Design Pattern: Strategy Interface (Strategy Pattern).
    Các lớp cài đặt cụ thể (SGD, Adam) triển khai chiến lược cập nhật trọng số
    dựa trên gradient tích lũy. Trainer giữ tham chiếu đến IOptimizer và gọi step()
    mà không cần phụ thuộc vào thuật toán cụ thể.
    """

    def __init__(self, lr: float) -> None:
        """Khởi tạo thuật toán tối ưu với tốc độ học (learning rate).

        Args:
            lr: Tốc độ học của thuật toán, yêu cầu lr > 0 (Rule R7).

        Raises:
            InvalidConfigError: Nếu lr <= 0.
        """
        # Rule R7: Kiểm tra learning rate > 0
        if lr <= 0:
            raise InvalidConfigError(f"Learning rate phải lớn hơn 0, nhận được lr={lr}")
        self._lr: float = lr

    @property
    def lr(self) -> float:
        """Tốc độ học hiện tại của bộ tối ưu."""
        return self._lr

    @abstractmethod
    def step(self, params: Sequence[Parameter]) -> None:
        """Cập nhật giá trị các tham số dựa trên gradient tích lũy.

        Args:
            params: Danh sách các Parameter cần cập nhật.
        """
        ...

    def zero_grad(self, params: Sequence[Parameter]) -> None:
        """Xóa trắng gradient của toàn bộ các tham số truyền vào.

        Args:
            params: Danh sách các Parameter cần xóa gradient.
        """
        for param in params:
            param.zero_grad()
