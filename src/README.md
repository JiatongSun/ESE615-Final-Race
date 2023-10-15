ESE615 F1tenth Team 05 Final Project

Instructions:
1. Put maps(.png, .pgm, etc) and its configuration(.yaml) under `maps`
2. Edit `config/params.yaml` to include correct map name
3. Run `./scripts/populate.sh` to populate data files into subdirectories
4. Run `python3 trajectory_generator/path_generator.py` to generate track data from image
5. Run `python3 trajectory_generator/main_globaltraj.py` to generate race line file
6. Run `./scripts/populate.sh` again to populate latest data files
7. Run `ros2 run pure_pursuit pure_pursuit_node.py` as a race example