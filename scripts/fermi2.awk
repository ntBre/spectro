#!/usr/bin/awk -f

{
    printf "Fermi2::new(%5d,%5d,%5d),\n", $1-1, $2-1, $3-1
}
