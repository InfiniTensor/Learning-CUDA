define PLATFORM_INIT_ENV
true
endef

exec: ./runTester
	srun --partition=mt --nodes=1 --gres=gpu:mt:2 --ntasks=1 --cpus-per-task=16 --mem=256G --time=00:20:00 ./runTester

