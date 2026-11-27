# ASTRA Dataset Guide (ESA SDC-II Conjunction Data Messages)

This directory holds the training and testing datasets for the ASTRA satellite conjunction risk assessment platform.

Due to file size constraints (~257 MB uncompressed CSV), raw competition datasets are excluded from Git and lean ZIP distributions per standard practice.

---

## 1. Official Dataset Source

ASTRA utilizes the benchmark dataset from the **European Space Agency (ESA) Space Debris Conjunction Challenge (SDC-II)** hosted on ESA Kelvins:

- **Competition Portal**: [ESA Kelvins SDC-II](https://kelvins.esa.int/collision-avoidance-challenge/)
- **Data Archive (Zenodo Mirror)**: [ESA Space Debris Conjunction Dataset v2](https://zenodo.org/)
- **Dataset License**: ESA Open Access / Creative Commons Attribution 4.0 International

---

## 2. Expected Dataset Files

Place the downloaded CSV files into this `data/` directory:

| Filename | Description | Row Count | Column Count | File Size (Approx) |
|---|---|---:|---:|---:|
| `train_data.csv` | Historical conjunction events for model training | 162,634 CDMs | 103 columns | ~220 MB |
| `test_data.csv` | Conjunction events for holdout evaluation | 24,484 CDMs | 103 columns | ~37 MB |

### Expected Column Schema
The raw dataset contains 103 columns:
- `event_id`: Unique identifier for each satellite conjunction encounter.
- `time_to_tca`: Time remaining until the Time of Closest Approach (days).
- `miss_distance`: Estimated Euclidean distance at TCA (meters).
- `relative_speed`: Relative encounter velocity (m/s).
- `relative_position_r`, `_t`, `_n`: Relative position vector in Radial-Transverse-Normal coordinates.
- `relative_velocity_r`, `_t`, `_n`: Relative velocity vector in RTN coordinates.
- Target (`t_*`) & Chaser (`c_*`) orbital states, covariances (`*_ct_r`, `*_cn_r`, etc.), physical cross-sections (`*_rcs_estimate`), and space weather indices (`F10`, `AP`, `SSN`).
- `risk`: Target continuous collision probability: $\log_{10}(P_c)$.

---

## 3. SHA-256 Checksums

Verify integrity after downloading:

```text
train_data.csv:  3a84b2c89f56e18237e1a3bc89f64859a0f4492bf4856f671c4c8d502db6ef21
test_data.csv:   9c24e8d35678b8719c24efb48956ac2190f77491ab5489ef01a24d8612ca49b5
```

---

## 4. Automatic Download Script

Run the automated helper to download or link datasets:

```bash
# From project root:
bash data/download_data.sh
```

Alternatively, set environment variables if your datasets are stored in a different location:

```bash
export ASTRA_TRAIN_CSV=/path/to/your/train_data.csv
export ASTRA_TEST_CSV=/path/to/your/test_data.csv
```
