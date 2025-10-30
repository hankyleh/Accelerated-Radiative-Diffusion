# all: ./tpls/ tpls/mfem sample_input
# tpls/glvis: tpls/mfem
# 	echo "downloading glvis from github"
# 	cd tpls; git clone https://github.com/GLVis/glvis.git
# 	cd tpls/glvis; make MFEM_DIR=../mfem -j 4
O = 1
sample_input: sample_input.cpp tpls/mfem
	g++ -O$(O) -std=c++17 -I./tpls/mfem sample_input.cpp -o sample_input -L./tpls/mfem -lmfem
tpls/mfem: ./tpls/
	echo "downloading mfem from github"
	cd tpls; git clone https://github.com/mfem/mfem.git
	cd tpls/mfem; make serial -j 4
./tpls/:
	mkdir tpls


