TESTFLAGS = -- --nocapture --test-threads=1
ARGS =

ifeq ($(ARGS),sym)
TESTFLAGS += --include-ignored _sym
else ifeq ($(ARGS),all)
TESTFLAGS += --include-ignored
else
TESTARGS = $(ARGS)
endif

test:
	RUST_BACKTRACE=1 cargo test ${TESTFLAGS} ${TESTARGS}

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
