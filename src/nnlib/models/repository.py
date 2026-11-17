"""Giao diện trừu tượng IModelRepository phục vụ lưu trữ và phục hồi mô hình."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nnlib.layers.base import Layer


class IModelRepository(ABC):
    """Giao diện trừu tượng quản lý việc lưu và nạp mô hình mạng nơ-ron.

    Vai trò trong Kiến trúc: Repository Interface (DIP).
    Tầng Lõi (nnlib) chỉ phụ thuộc vào giao diện này, trong khi việc lưu trữ
    cụ thể (cấu trúc JSON + trọng số NPZ) được cài đặt tại tầng hạ tầng
    (nnlib/io/NpzModelRepository).
    Nhờ đó, thay đổi phương thức lưu trữ không làm ảnh hưởng đến mã nguồn tầng Lõi.
    """

    @abstractmethod
    def save(self, model: Layer, path: str) -> None:
        """Lưu trữ cấu trúc và tham số trọng số của mô hình xuống đường dẫn chỉ định.

        Args:
            model: Đối tượng Layer (thường là Sequential) cần lưu.
            path: Đường dẫn tới file/thư mục lưu trữ (không chứa đuôi mở rộng).
        """
        ...

    @abstractmethod
    def load(self, path: str) -> Layer:
        """Khôi phục đối tượng Layer từ đường dẫn lưu trữ.

        Args:
            path: Đường dẫn tới file/thư mục lưu trữ đã lưu trước đó.

        Returns:
            Đối tượng Layer hợp lệ đã được nạp đầy đủ cấu trúc và trọng số.
        """
        ...
