ARGS =

TESTFLAGS = -- --nocapture

TARGET = x86_64-unknown-linux-gnu

test:
	cargo test --workspace ${TESTFLAGS} ${ARGS}

bench:
	cargo +nightly bench ${TESTFLAGS} ${ARGS}

clippy:
	cargo clippy --workspace --tests

clean:
	cargo clean

%.pdf : %.gv
	dot -Tpdf $< -o $@

%.svg : %.gv
	dot -Tsvg $< -o $@

flow: flow/asym.pdf

build:
	RUSTFLAGS='-C target-feature=+crt-static' cargo build -p spectro_bin \
		    --release --target $(TARGET)
woods: build
	scp -C ${BASE}/target/$(TARGET)/release/spectro_bin \
                'woods:bin/rspectro'$(ALPHA)

eland: build
	scp -C ${BASE}/target/$(TARGET)/release/spectro_bin \
                'eland:bin/rspectro'

.PHONY: install
install:
	cargo install --path spectro_bin

poly.test:
	cargo run --features polyad -p spectro_bin spectro/testfiles/c2h4/spectro.in

#############
# PROFILING #
#############

BASE = .

profile = RUSTFLAGS='-g' cargo build --release --bin $(1); \
        valgrind --tool=callgrind --callgrind-out-file=callgrind.out    \
                --collect-jumps=yes --simulate-cache=yes                \
                ${BASE}/target/release/$(1) spectro/testfiles/c3h2/spectro.in

profile:
	$(call profile,prof)
