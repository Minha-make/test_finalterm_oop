"""Giao diện trừu tượng ILossFunction cho các hàm mất mát."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nnlib.tensor import Tensor


class ILossFunction(ABC):
    """Giao diện trừu tượng cho các hàm mất mát (Loss Function).

    Vai trò trong Design Pattern: Strategy Interface (Strategy Pattern).
    Các lớp cài đặt cụ thể (MSELoss, CrossEntropyLoss) sẽ hiện thực hóa chiến lược
    tính toán mất mát và gradient tương ứng, cho phép Trainer hoán đổi chiến lược
    linh hoạt theo cấu hình mà không phải sửa đổi mã nguồn thuật toán huấn luyện.
    """

    @abstractmethod
    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Tính toán giá trị mất mát vô hướng trung bình trên lô dữ liệu.

        Args:
            y_pred: Tensor dự đoán của mô hình kích thước (N, C).
            y_true: Tensor nhãn thực tế dạng one-hot kích thước (N, C).

        Returns:
            Giá trị mất mát vô hướng (scalar float).
        """
        ...

    @abstractmethod
    def backward(self) -> Tensor:
        """Tính đạo hàm của hàm mất mát đối với giá trị dự đoán dL/dy_pred.

        QUY ƯỚC BẮT BUỘC (Mục 6):
        Gradient trả về từ backward() PHẢI ĐÃ ĐƯỢC CHIA CHO N (kích thước batch),
        để các Layer phía trước không cần chia lại N khi cập nhật trọng số.

        Returns:
            Tensor gradient dL/dy_pred có cùng kích thước với y_pred.
        """
        ...
