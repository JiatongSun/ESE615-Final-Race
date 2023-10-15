#!/bin/bash
PACKAGE=(mpc opponent_predictor trajectory_generator pure_pursuit rrt dummy_car lane_follow)
SUB_DIR=(config maps csv)

# if there are new output csv, put it under root csv folder
if [ -d "./trajectory_generator/outputs" ]; then
  cp -r ./trajectory_generator/outputs/* ./csv/
fi

# loop all packages
for p in "${PACKAGE[@]}"; do
  # if package does not exist, move on
  if [ ! -d "./$p/" ]; then
    echo "Skipping $p package"
    continue
  fi

  # otherwise, build necessary subdirectories
  echo "Copying into $p package..."
  for s in "${SUB_DIR[@]}"; do
    if [ -d "./$p/$s/" ]; then
      rm -rf "./$p/$s/"
    fi
    mkdir -p "./$p/$s/"
    cp -r ./"$s"/* "./$p/$s/"
  done
done
