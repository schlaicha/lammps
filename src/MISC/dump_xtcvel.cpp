/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Naveen Michaud-Agrawal (Johns Hopkins U)
                         open-source XDR routines from
                           Frans van Hoesel (http://md.chem.rug.nl/hoesel)
                           are included in this file
                         Axel Kohlmeyer (Temple U)
                           port to platforms without XDR support
                           added support for unwrapped trajectories
                           support for groups
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "dump_xtcvel.h"
#include "domain.h"
#include "atom.h"
#include "update.h"
#include "group.h"
#include "output.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define EPS 1e-5
#define XTC_MAGIC 1995

#define MYMIN(a,b) ((a) < (b) ? (a) : (b))
#define MYMAX(a,b) ((a) > (b) ? (a) : (b))

int xdropen(XDR *, const char *, const char *);
int xdrclose(XDR *);
void xdrfreebuf();
int xdr3dfcoord(XDR *, float *, int *, float *);

/* ---------------------------------------------------------------------- */

DumpXTCvel::DumpXTCvel(LAMMPS *lmp, int narg, char **arg) : Dump(lmp, narg, arg), 
  coords(NULL)
{
  if (narg != 5) error->all(FLERR,"Illegal dump xtc command");
  if (binary || compressed || multifile || multiproc)
    error->all(FLERR,"Invalid dump xtc filename");

  size_one = 3;
  sort_flag = 1;
  sortcol = 0;
  format_default = NULL;
  flush_flag = 0;
  precision = 1000.0;

  // allocate global array for atom coords

  bigint n = group->count(igroup);
  if (n > static_cast<bigint>(MAXSMALLINT/3/sizeof(float)))
    error->all(FLERR,"Too many atoms for dump xtc");
  natoms = static_cast<int> (n);

  memory->create(coords,3*natoms,"dump:coords");

  // sfactor = conversion of coords to XTC units
  // tfactor = conversion of simulation time to XTC units
  // GROMACS standard is nanometers and picoseconds

  sfactor = 1.e2 / (force->angstrom / force->femtosecond); // velocity in nm/ps
  tfactor = 0.001 / force->femtosecond;

  // in reduced units we do not scale anything
  if (strcmp(update->unit_style,"lj") == 0) {
    sfactor = tfactor = 1.0;
    if (comm->me == 0)
      error->warning(FLERR,"No automatic unit conversion to XTC file "
                     "format conventions possible for units lj");
  }

  openfile();
  nevery_save = 0;
  ntotal = 0;
}

/* ---------------------------------------------------------------------- */

DumpXTCvel::~DumpXTCvel()
{
  memory->destroy(coords);

  if (me == 0) {
    xdrclose(&xd);
    xdrfreebuf();
  }
}

/* ---------------------------------------------------------------------- */

void DumpXTCvel::init_style()
{
  if (sort_flag == 0 || sortcol != 0)
    error->all(FLERR,"Dump xtc requires sorting by atom ID");

  // check that flush_flag is not set since dump::write() will use it

  if (flush_flag) error->all(FLERR,"Cannot set dump_modify flush for dump xtc");

  // check that dump frequency has not changed and is not a variable

  int idump;
  for (idump = 0; idump < output->ndump; idump++)
    if (strcmp(id,output->dump[idump]->id) == 0) break;
  if (output->every_dump[idump] == 0)
    error->all(FLERR,"Cannot use variable every setting for dump xtc");

  if (nevery_save == 0) nevery_save = output->every_dump[idump];
  else if (nevery_save != output->every_dump[idump])
    error->all(FLERR,"Cannot change dump_modify every for dump xtc");
}

/* ---------------------------------------------------------------------- */

void DumpXTCvel::openfile()
{
  // XTC maintains it's own XDR file ptr
  // set fp to NULL so parent dump class will not use it

  fp = NULL;
  if (me == 0)
    if (xdropen(&xd,filename,"w") == 0) error->one(FLERR,"Cannot open dump file");
}

/* ---------------------------------------------------------------------- */

void DumpXTCvel::write_header(bigint nbig)
{
  if (nbig > MAXSMALLINT) error->all(FLERR,"Too many atoms for dump xtc");
  int n = nbig;
  if (update->ntimestep > MAXSMALLINT)
    error->one(FLERR,"Too big a timestep for dump xtc");
  int ntimestep = update->ntimestep;

  // all procs realloc coords if total count grew

  if (n != natoms) {
    natoms = n;
    memory->destroy(coords);
    memory->create(coords,3*natoms,"dump:coords");
  }

  // only proc 0 writes header

  if (me != 0) return;

  int tmp = XTC_MAGIC;
  xdr_int(&xd,&tmp);
  xdr_int(&xd,&n);
  xdr_int(&xd,&ntimestep);
  float time_value = ntimestep * tfactor * update->dt;
  xdr_float(&xd,&time_value);

  // cell basis vectors
  if (domain->triclinic) {
    float zero = 0.0;
    float xdim = sfactor * (domain->boxhi[0] - domain->boxlo[0]);
    float ydim = sfactor * (domain->boxhi[1] - domain->boxlo[1]);
    float zdim = sfactor * (domain->boxhi[2] - domain->boxlo[2]);
    float xy = sfactor * domain->xy;
    float xz = sfactor * domain->xz;
    float yz = sfactor * domain->yz;

    xdr_float(&xd,&xdim); xdr_float(&xd,&zero); xdr_float(&xd,&zero);
    xdr_float(&xd,&xy  ); xdr_float(&xd,&ydim); xdr_float(&xd,&zero);
    xdr_float(&xd,&xz  ); xdr_float(&xd,&yz  ); xdr_float(&xd,&zdim);
  } else {
    float zero = 0.0;
    float xdim = sfactor * (domain->boxhi[0] - domain->boxlo[0]);
    float ydim = sfactor * (domain->boxhi[1] - domain->boxlo[1]);
    float zdim = sfactor * (domain->boxhi[2] - domain->boxlo[2]);

    xdr_float(&xd,&xdim); xdr_float(&xd,&zero); xdr_float(&xd,&zero);
    xdr_float(&xd,&zero); xdr_float(&xd,&ydim); xdr_float(&xd,&zero);
    xdr_float(&xd,&zero); xdr_float(&xd,&zero); xdr_float(&xd,&zdim);
  }
}

/* ---------------------------------------------------------------------- */

void DumpXTCvel::pack(tagint *ids)
{
  int m,n;

  tagint *tag = atom->tag;
  double **x = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  m = n = 0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      buf[m++] = sfactor*x[i][0];
      buf[m++] = sfactor*x[i][1];
      buf[m++] = sfactor*x[i][2];
      ids[n++] = tag[i];
    }
}

/* ---------------------------------------------------------------------- */

void DumpXTCvel::write_data(int n, double *mybuf)
{
  // copy buf atom coords into global array

  int m = 0;
  int k = 3*ntotal;
  for (int i = 0; i < n; i++) {
    coords[k++] = mybuf[m++];
    coords[k++] = mybuf[m++];
    coords[k++] = mybuf[m++];
    ntotal++;
  }

  // if last chunk of atoms in this snapshot, write global arrays to file

  if (ntotal == natoms) {
    write_frame();
    ntotal = 0;
  }
}

/* ---------------------------------------------------------------------- */

int DumpXTCvel::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"unwrap") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal dump_modify command");
    else error->all(FLERR,"Illegal dump_modify command");
    return 2;
  } else if (strcmp(arg[0],"precision") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal dump_modify command");
    precision = force->numeric(FLERR,arg[1]);
    if ((fabs(precision-10.0) > EPS) && (fabs(precision-100.0) > EPS) &&
        (fabs(precision-1000.0) > EPS) && (fabs(precision-10000.0) > EPS) &&
        (fabs(precision-100000.0) > EPS) &&
        (fabs(precision-1000000.0) > EPS))
      error->all(FLERR,"Illegal dump_modify command");
    return 2;
  } else if (strcmp(arg[0],"sfactor") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal dump_modify command");
    sfactor = force->numeric(FLERR,arg[1]);
    if (sfactor <= 0.0)
      error->all(FLERR,"Illegal dump_modify sfactor value (must be > 0.0)");
    return 2;
  } else if (strcmp(arg[0],"tfactor") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal dump_modify command");
    tfactor = force->numeric(FLERR,arg[1]);
    if (tfactor <= 0.0)
      error->all(FLERR,"Illegal dump_modify tfactor value (must be > 0.0)");
    return 2;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory in buf and global coords array
------------------------------------------------------------------------- */

bigint DumpXTCvel::memory_usage()
{
  bigint bytes = Dump::memory_usage();
  bytes += memory->usage(coords,natoms*3);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void DumpXTCvel::write_frame()
{
  xdr3dfcoord(&xd,coords,&natoms,&precision);
}

// ----------------------------------------------------------------------
// C functions that create GROMOS-compatible XDR files are already
// included from dump_xtc.cpp
// ----------------------------------------------------------------------
