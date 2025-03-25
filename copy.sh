#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
scp "$1" pivotal:
ssh pivotal "sudo mv $1 /var/www/html/"
echo http://3.10.22.91/"$1"
