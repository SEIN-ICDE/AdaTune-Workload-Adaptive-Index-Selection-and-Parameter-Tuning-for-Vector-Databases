"""
configs/sift.py  —  SIFT-128 (raw, L2)
"""

CFG = {
    "name":       "sift",
    "ds_path":    "/home/sein/Vector_DBMS/dataset/sift-128-euclidean.hdf5",
    "metric":     "L2",
    "index_types": ["HNSW", "SCANN"],
    "result_dir": "results",

    "milvus_uri":      "http://localhost:19530",
    "health_url":      "http://localhost:9091/healthz",
    "container_name":  "milvus-standalone",
    "user_yaml_path":  "/home/sein/user.yaml",

    "index": {
        "HNSW": {
            "build": {
                "M":              {"range": (4,   64),  "scale": "log", "dtype": int},
                "efConstruction": {"range": (40,  400), "scale": "log", "dtype": int},
            },
            "search": {
                "ef":             {"range": (16,  512), "scale": "log", "dtype": int},
            },
            "build_default_grid": [
                {"M": 16, "efConstruction": 100},
                {"M": 32, "efConstruction": 200},
                {"M": 48, "efConstruction": 300},
            ],
        },
        "SCANN": {
            "build": {
                "nlist":          {"range": (256, 4096), "scale": "log", "dtype": int},
            },
            "search": {
                "nprobe":         {"range": (1,   512),  "scale": "log", "dtype": int},
            },
            "build_default_grid": [
                {"nlist": 512},
                {"nlist": 1024},
                {"nlist": 2048},
            ],
        },
    },
}
