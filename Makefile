ARGS =

TESTFLAGS = -- --nocapture --test-threads=1

test:
	cargo test ${TESTFLAGS} ${ARGS}

%.pdf : %.gv
	dot -Tpdf $< -o $@

flow: flow/asym.pdf

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
