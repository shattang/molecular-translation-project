if ! [ -z "$(git status --porcelain)" ]; then
  RED='\033[0;31m'
  NC='\033[0m' # No Color
  printf "${RED}Git repo is not clean${NC}\n"  
  exit 1  
fi

mv -f code.zip code.backup.zip
git rev-parse HEAD > version.info
zip code.zip version.info
zip code.zip *.py
zip code.zip settings/*
zip code.zip scripts/*
zip code.zip model1/*.py

