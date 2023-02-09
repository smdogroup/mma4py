default:
	python3 setup.py build_ext --inplace

clean:
	rm -rf build mma4py.egg-info mma4py/*.so