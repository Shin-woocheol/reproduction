import os
import glob
import re
import numpy as np
import argparse

def parse_line(line):
    """
    한 줄을 파싱하여 (time, eecbs, soc, ms, cost_time) 튜플을 반환합니다.
    첫 번째 줄은 "Hungarian time:"를 사용하고, 이후는 "TIMECOST:"를 사용한다고 가정합니다.
    예시:
        "0 EECBS_MAKESPAN: 31 SOC(A*): 213 MS(A*): 31 Hungarian time: 0.4163"
        "5 EECBS_MAKESPAN: 31 SOC(A*): 225 MS(A*): 31 TIMECOST: 0.9260"
    """
    # 정규표현식: 맨 앞 숫자, 이후 각 metric과 값. 시간 관련 항목은 "Hungarian time" 또는 "TIMECOST" 둘 다 잡음.
    pattern = r"^(\d+)\s+EECBS_MAKESPAN:\s+(\d+)\s+SOC\(A\*\):\s+(\d+)\s+MS\(A\*\):\s+(\d+)\s+(?:Hungarian time|TIMECOST):\s+([\d.]+)"
    m = re.match(pattern, line.strip())
    if m:
        time_val = int(m.group(1))
        eecbs = float(m.group(2))
        soc = float(m.group(3))
        ms = float(m.group(4))
        cost_time = float(m.group(5))
        return time_val, eecbs, soc, ms, cost_time
    else:
        return None

def process_logs(log_folder):
    # dictionary: key -> time (정수), value -> list of (eecbs, soc, ms, cost_time)
    metrics_by_time = {}

    # log 파일 찾기 (예: scenario_*.txt)
    file_pattern = os.path.join(log_folder, "scenario_*.txt")
    files = glob.glob(file_pattern)
    if not files:
        print("No log files found in", log_folder)
        return

    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                # 마지막 줄에 [..Solution] 같은 형식이 있다면 무시합니다.
                if line.strip().startswith("["):
                    continue
                # 첫 글자가 숫자인지 확인
                if not line.strip() or not line.strip()[0].isdigit():
                    continue
                parsed = parse_line(line)
                if parsed is None:
                    print("Warning: couldn't parse line:", line.strip())
                    continue
                time_val, eecbs, soc, ms, cost_time = parsed
                if time_val not in metrics_by_time:
                    metrics_by_time[time_val] = []
                metrics_by_time[time_val].append((eecbs, soc, ms, cost_time))
    
    # 각 시간별 평균 계산
    summary_lines = []
    header = "Time EECBS_MAKESPAN SOC(A*) MS(A*) COST_TIME"
    summary_lines.append(header)
    
    # 정렬해서 처리 (예: 0, 5, 10, ...)
    for t in sorted(metrics_by_time.keys()):
        data = np.array(metrics_by_time[t])  # shape: (n, 4)
        mean_vals = data.mean(axis=0)
        # 출력 포맷 조정 (정수형과 실수형 구분)
        line = f"{t} EECBS_MAKESPAN: {mean_vals[0]:.2f} SOC(A*): {mean_vals[1]:.2f} MS(A*): {mean_vals[2]:.2f} COST_TIME: {mean_vals[3]:.4f}"
        summary_lines.append(line)
    
    # summary.txt 파일 저장
    summary_path = os.path.join(log_folder, "summary.txt")
    with open(summary_path, "w") as f:
        for line in summary_lines:
            f.write(line + "\n")
    
    print("Summary written to", summary_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario_fol', type = str, default = 'test_scenario_seed7777')
    args = parser.parse_args()
    log_folder = os.path.join(".", "LNS_result", args.scenario_fol)
    process_logs(log_folder)

#python summary.py --scenario_fol "test_scenario_seed7777"