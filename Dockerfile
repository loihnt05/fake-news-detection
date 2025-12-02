FROM postgres:16
# Dùng bản Debian (mặc định) cài đặt cực dễ, không lỗi Clang
RUN apt-get update && apt-get install -y \
    git make build-essential postgresql-server-dev-16 \
    && cd /tmp \
    && git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector