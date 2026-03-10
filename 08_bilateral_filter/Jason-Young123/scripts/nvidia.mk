define PLATFORM_INIT_ENV
[ -f /data/shared/miniconda3/etc/profile.d/conda.sh ] && . /data/shared/miniconda3/etc/profile.d/conda.sh || true
endef

exec: ./runTester
	./runTester

