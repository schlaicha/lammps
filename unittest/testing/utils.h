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
#ifndef TEST_EXTENSIONS__H
#define TEST_EXTENSIONS__H

#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

static void delete_file(const std::string &filename)
{
    remove(filename.c_str());
}

static size_t count_lines(const std::string &filename)
{
    std::ifstream infile(filename);
    std::string line;
    size_t nlines = 0;

    while (std::getline(infile, line))
        ++nlines;

    return nlines;
}

static bool equal_lines(const std::string &fileA, const std::string &fileB)
{
    std::ifstream afile(fileA);
    std::ifstream bfile(fileB);
    std::string lineA, lineB;

    while (std::getline(afile, lineA)) {
        if (!std::getline(bfile, lineB)) return false;
        if (lineA != lineB) return false;
    }

    return true;
}

static std::vector<std::string> read_lines(const std::string &filename)
{
    std::vector<std::string> lines;
    std::ifstream infile(filename);
    std::string line;

    while (std::getline(infile, line))
        lines.push_back(line);

    return lines;
}

static bool file_exists(const std::string &filename)
{
    struct stat result;
    return stat(filename.c_str(), &result) == 0;
}

#define ASSERT_FILE_EXISTS(NAME) ASSERT_TRUE(file_exists(NAME))
#define ASSERT_FILE_EQUAL(FILE_A, FILE_B) ASSERT_TRUE(equal_lines(FILE_A, FILE_B))

#endif
