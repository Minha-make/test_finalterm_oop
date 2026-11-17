# HƯỚNG DẪN ĐÓNG GÓP & QUY TRÌNH PHỐI HỢP DỰ ÁN (CONTRIBUTING)
## Dự án: Thư viện mạng nơ-ron OOP & Ứng dụng nhận diện chữ số viết tay (MNIST)

Tài liệu này quy định chi tiết quy trình phối hợp phát triển, phân công trách nhiệm giữa hai thành viên, quy chuẩn Git, quy định về hợp đồng giao tiếp, tiêu chuẩn mã nguồn và kiểm thử.

---

## 1. Phân công trách nhiệm giữa hai thành viên

Dự án được chia thành 2 phân hệ độc lập tương ứng với 2 thành viên:

| Hạng mục | Người A (Phân hệ Lõi `nnlib`) | Người B (Huấn luyện & Ứng dụng `digitapp`) |
|:---|:---|:---|
| **Thư mục phụ trách** | `src/nnlib/tensor.py`, `exceptions.py`<br>`src/nnlib/layers/`<br>`src/nnlib/initializers/`<br>`src/nnlib/losses/`<br>`src/nnlib/models/`<br>`src/nnlib/io/` | `src/nnlib/optim/`<br>`src/nnlib/training/`<br>`src/digitapp/data/`<br>`src/digitapp/metrics/`<br>`src/digitapp/config/`, `common/`<br>`src/digitapp/controllers/` (trừ ModelController)<br>`src/digitapp/cli/`, `gui/`, `predictor/` |
| **Use Cases** | **UC1 – UC4**: Xây dựng mô hình, Forward, Gradient check, Lưu/Nạp mô hình | **UC5 – UC8**: Huấn luyện, Đánh giá test, Nhận diện vẽ tay GUI, Nhận diện file CLI |
| **Hàm Controller** | `ModelController` (`build_model`, `forward`, `check_gradients`, `save_model`, `load_model`) | `TrainController`, `EvaluateController`, `RecognitionController` |
| **Quy tắc nghiệp vụ** | **R1 – R6**: Ràng buộc Dense 2D, Cache backward, Dropout inverted, Ổn định số Softmax/CrossEntropy, Khớp chiều Dense, Kiểm tra file nạp | **R7 – R12**: Siêu tham số Adam, EarlyStopping, Tách train/val không shuffle val, Tiền xử lý nét vẽ, Ngưỡng tin cậy, Canvas trống |
| **Design Patterns** | **Composite Pattern** (`Layer`, `Dense`, `Sequential`)<br>**Factory Pattern** (`LayerFactory` có Registry) | **Strategy Pattern** (`Trainer`, `IOptimizer`, `ILossFunction`)<br>**Observer Pattern** (`Trainer`, `ICallback`, `EarlyStopping`,...) |
| **Bài mở rộng** | Thêm lớp `LeakyReLU`, `Flatten` (commit chỉ thêm file mới) | Thêm `RMSprop`, `CsvLoggerCallback` (commit chỉ thêm file mới) |
| **Kiểm thử đơn vị** | Ca kiểm thử **T01 – T12** ($\ge 12$ tests) | Ca kiểm thử **T13 – T24** ($\ge 12$ tests) |
| **Tài liệu & Đóng gói** | Viết chương 1, 2 Báo cáo; tổng hợp sơ đồ UML | Viết chương 3, 4, 5 Báo cáo; đóng gói PyInstaller (`resource_path`) |

---

## 2. Hợp đồng giao tiếp (Contract Interfaces - Tag `v0.1-contract`)

Hai phân hệ chỉ tương tác với nhau thông qua **5 giao diện hợp đồng** được quy định tại Mục 6 của kế hoạch và Mục 2 của `ARCHITECTURE_RULES.md`:
1. `Tensor` & `Parameter` (`src/nnlib/tensor.py`)
2. `Layer` (`src/nnlib/layers/base.py`)
3. `ILossFunction` (`src/nnlib/losses/base.py`)
4. `IOptimizer` (`src/nnlib/optim/base.py`)
5. `IModelRepository` (`src/nnlib/models/repository.py`)

### Nguyên tắc phối hợp độc lập (Decoupling):
- **Phân hệ B không chờ A**: Trong giai đoạn tuần 1–2, Người B sử dụng `FakeLayer` (được định nghĩa trong `tests/fakes/`) để kiểm thử và hoàn thiện `SGD`, `Adam`, `Trainer` mà không cần đợi các Layer thực sự từ Người A.
- **Phân hệ A không chờ B**: Người A sử dụng một vòng lặp huấn luyện độc lập ngắn (~15 dòng) trong test để kiểm chứng mạng học được bài toán XOR mà không cần đợi `Trainer` từ Người B.
- **Quy tắc thay đổi Hợp đồng**: Chữ ký của 5 interface trên được chốt tại tuần 1 và gắn tag `v0.1-contract`. Tuyệt đối không tự ý thay đổi chữ ký. Nếu bắt buộc phải thay đổi, phải tạo Pull Request có nhãn `contract` và được **cả hai thành viên cùng phê duyệt**.

---

## 3. Quy trình làm việc với Git & Phân nhánh (Mục 10)

Để đảm bảo đáp ứng chỉ tiêu $\ge 24$ commit có ý nghĩa ($\ge 12$ commit mỗi người) và quản lý mã nguồn chuyên nghiệp:

### 3.1. Cấu trúc nhánh (Branching Model)
- `main`: Nhánh ổn định, chỉ chứa các bản phát hành có gắn tag phiên bản (`v0.1-contract`, `v0.2`, ..., `v1.0`).
- `develop`: Nhánh tích hợp chính của dự án. Mọi tính năng sau khi kiểm thử đạt yêu cầu sẽ được gộp vào đây.
- `feature/a-<tên-chủ-đề>`: Nhánh làm việc riêng của Người A (ví dụ: `feature/a-dense-layer`, `feature/a-composite`).
- `feature/b-<tên-chủ-đề>`: Nhánh làm việc riêng của Người B (ví dụ: `feature/b-optimizer-sgd`, `feature/b-tkinter-gui`).

### 3.2. Quy trình gộp nhánh (Pull Request & Merge)
1. Mỗi tính năng được hoàn thiện trên nhánh `feature/*`.
2. Tạo Pull Request (PR) trỏ vào nhánh `develop`.
3. Thành viên còn lại có trách nhiệm review code và phê duyệt trong vòng 24 giờ.
4. Gộp PR bằng **merge commit** (`git merge --no-ff`). **TUYỆT ĐỐI KHÔNG DÙNG SQUASH MERGE** để bảo toàn đầy đủ lịch sử commit của từng cá nhân.

### 3.3. Quy chuẩn thông điệp Commit (Conventional Commits)
Thông điệp commit phải ngắn gọn, rõ ràng theo định dạng:
`<loại>(<phạm vi>): <nội dung tóm tắt>`

Các loại commit chuẩn:
- `feat`: Tính năng mới (ví dụ: `feat(layers): thêm Dropout layer với quy tắc R3`)
- `test`: Thêm hoặc chỉnh sửa ca kiểm thử (ví dụ: `test(optim): kiểm tra Adam step và bias correction R7`)
- `docs`: Cập nhật tài liệu (ví dụ: `docs: cập nhật ARCHITECTURE_RULES.md`)
- `fix`: Sửa lỗi mã nguồn (ví dụ: `fix(losses): kẹp xác suất chống log(0) theo R4`)
- `refactor`: Tái cấu trúc mã nguồn mà không làm thay đổi hành vi

### 3.4. Định danh tác giả Git
Mỗi thành viên phải cấu hình đúng `user.name` và `user.email` riêng trên máy tính của mình trước khi commit để lệnh thống kê đóng góp `git shortlog -sn` phản ánh chính xác kết quả:
```bash
git config user.name "Họ và Tên"
git config user.email "email_cua_ban@example.com"
```

---

## 4. Quy chuẩn viết mã (Coding Standards)

Mọi mã nguồn đóng góp vào dự án bắt buộc phải tuân theo các quy định trong [ARCHITECTURE_RULES.md](file:///c:/Users/Hi/Desktop/Documents/OOP/test_finalterm/neuralnet-oop/ARCHITECTURE_RULES.md):

1. **Tuân thủ kiến trúc 4 tầng**:
   - `Presentation` (`cli`, `gui`) $\to$ `Controllers` $\to$ `Domain Core` (`nnlib`, core app) $\to$ `Infrastructure` (`io`, `config`).
   - Lời gọi chỉ đi một chiều từ trên xuống; không có import ngược.
2. **Quy tắc đóng gói (Encapsulation)**:
   - Toàn bộ biến thành viên lưu trạng thái bên trong lớp phải có tiền tố `_ten_bien`.
   - Cung cấp `@property` và setter kiểm tra dữ liệu nếu cho phép truy xuất/thay đổi từ bên ngoài.
3. **Quy chuẩn PEP 8, Type Hints & Docstrings**:
   - Định dạng code theo chuẩn Black (độ dài dòng 88 ký tự). Kiểm tra bằng Ruff.
   - 100% hàm/phương thức phải có đầy đủ type annotations.
   - 100% module, class và public method phải có docstring rõ ràng bằng tiếng Việt.
   - Mọi lớp tham gia Design Pattern phải ghi rõ: `Vai trò trong Design Pattern: <Role> (<Pattern>)`.
4. **Quy tắc nghiệp vụ (R1–R12)**:
   - Phương thức nào cài đặt quy tắc nghiệp vụ phải có comment `# Rule R<x>` ngay tại dòng xử lý.
5. **Xử lý tài nguyên và lỗi**:
   - Mọi đường dẫn file phải dùng hàm `resource_path(relative_path)` trong `src/digitapp/common/paths.py`. Tuyệt đối không hardcode đường dẫn tuyệt đối dạng `C:\...` hay `/home/...`.
   - Không nuốt lỗi (`except Exception: pass`). Tầng lõi chỉ raise ngoại lệ kế thừa `NNError`.

---

## 5. Quy trình Kiểm thử & Nghiệm thu (Testing & QA)

1. **Chạy kiểm thử**:
   Mọi PR trước khi gộp vào `develop` phải vượt qua toàn bộ test suite:
   ```bash
   pytest
   ```
2. **Gradient Check**:
   Mọi Layer mới có tham số hoặc hàm kích hoạt/mất mát mới đều phải vượt qua kiểm tra đạo hàm giải tích (`GradientChecker`) với sai số tương đối:
   $$\text{Relative Error} = \frac{|a - n|}{\max(|a| + |n|, 10^{-12})} < 10^{-5} \quad (\text{sử dụng } np.float64)$$
3. **Kiểm tra kiến trúc (Test I2)**:
   Bộ kiểm thử kiến trúc trong `tests/unit/test_architecture.py` sẽ tự động quét cây cú pháp (AST) của toàn bộ module để chứng minh không có câu lệnh `import` vi phạm kiến trúc 4 tầng.
