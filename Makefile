ARGS =

TESTFLAGS = -- --nocapture --test-threads=1

test:
	RUST_BACKTRACE=1 cargo test ${TESTFLAGS} ${ARGS}

#############
# PROFILING #
#############

BASE = .

profile = RUSTFLAGS='-g' cargo build --release --bin $(1); \
        valgrind --tool=callgrind --callgrind-out-file=callgrind.out    \
                --collect-jumps=yes --simulate-cache=yes                \
                ${BASE}/target/release/$(1)

profile:
	$(call profile,spectro)
