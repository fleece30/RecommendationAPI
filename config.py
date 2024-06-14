import os

DATA_DIR = os.getenv("DATA_DIR", "data")

OVERVIEWS_CSV_PATH = os.path.join(DATA_DIR, "Overviews.csv")
COSINE_SIM__PART_1_PATH = os.path.join(DATA_DIR, "cosine_sim_part_1.npz")
COSINE_SIM__PART_2_PATH = os.path.join(DATA_DIR, "cosine_sim_part_2.npz")
COSINE_SIM2_PATH = os.path.join(DATA_DIR, "cosine_sim2.npz")
INDICES_PATH = os.path.join(DATA_DIR, "indices.txt")