# Dữ liệu DSADS tại local
Đường dẫn chuẩn hoá nội bộ dự án: `data/raw` (đường dẫn tương đối tính từ gốc dự án).
Dữ liệu đã được đồng bộ trực tiếp vào `data/raw/` bao gồm 19 hoạt động (`a01` … `a19`), mỗi thư mục có 8 người (`p1` … `p8`), mỗi người có 60 đoạn (`s01.txt` … `s60.txt`).
Dữ liệu bộ nhớ đệm (cache `.npz`) được lưu tại `data/processed/dsads_cache.npz`.

## Kiểm tra
Từ thư mục dự án chạy:
```bash
python scripts/validate_data.py
```
Lệnh này chỉ kiểm tra dữ liệu và số mẫu theo split; không train hoặc đo metric trên test.
Kết quả hợp lệ được lưu trong reports/data_validation.json. Tập đầy đủ dự kiến có 9.120 mẫu; train 5.700, validation 1.140, test 2.280.
Nếu báo thiếu dữ liệu, không tiếp tục train trước khi xác minh nguyên nhân.

## Đổi vị trí
Sửa data_dir trong config.json nếu dữ liệu nằm ở vị trí khác. Ví dụ đặt dữ liệu trong data/raw/DSADS rồi dùng "data_dir": "data/raw/DSADS".
Không cần tải lại từ UCI nếu bản đã tải còn đầy đủ.

## Trạng thái tích hợp
Phiên Codex ngày 28/10/2026 bị hệ thống từ chối đọc thư mục dữ liệu, kể cả khi dùng quyền mở rộng. Vì thế chưa xác minh cấu trúc thực tế, số mẫu và chất lượng của bản đã giải nén.
Cần cho Codex quyền đọc thư mục, hoặc đặt bản dữ liệu vào vị trí đọc được và cập nhật config.json.
