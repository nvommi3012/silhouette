.DEFAULT_GOAL := all

MPICC = mpiCC
MPICFLAGS = -std=c++11
MPICOPTFLAGS = -O3 -g -lpng
MPILDFLAGS =

TARGETS = silhouette_mpi$(EXEEXT) silhouette_mpi_sr$(EXEEXT)

all: $(TARGETS)

DISTFILES += $(DEPS_COMMON)

silhouette_mpi$(EXEEXT): silhouette_mpi.cpp $(DEPS_COMMON)
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -I/data/apps/boost/1.57/include \
	    -o $@ silhouette_mpi.cpp $(MPILDFLAGS)

silhouette_mpi_sr$(EXEEXT): silhouette_mpi_sr.cpp $(DEPS_COMMON)
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -I/data/apps/boost/1.57/include \
	    -o $@ silhouette_mpi_sr.cpp $(MPILDFLAGS) -fopenmp

clean:
	rm -f $(TARGETS) 

# eof
