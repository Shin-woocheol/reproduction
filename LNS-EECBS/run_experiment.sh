#!/bin/bash
# run_experiment.sh
# 사용법:
#   ./run_experiments.sh <scenario_fol> <scenario_count> <parallel_count>
# chmod +x run_experiment.sh
# 예시:
#   ./run_experiment.sh test_scenario_seed7777 50 25

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <scenario_fol> <scenario_count> <parallel_count>"
    exit 1
fi

scenario_fol=$1
scenario_count=$2
parallel_count=$3
max_t=$4

echo "실험 시작: scenario folder = ${scenario_fol}, 시나리오 갯수 = ${scenario_count}, 동시에 실행할 프로세스 수 = ${parallel_count}"

# 백그라운드 작업 제어를 위한 카운터
running=0

for ((i=1; i<=scenario_count; i++)); do
    echo "시나리오 번호 ${i} 실행 중..."
    # main.py 실행 (각 시나리오마다 scenario_num을 지정)
    python lns_init.py --scenario_fol "$scenario_fol" --scenario_num "$i" --max_t $max_t&
    
    ((running++))
    # 1초 대기
    sleep 1

    # 지정한 병렬 실행 개수에 도달하면, 백그라운드 프로세스가 모두 끝날 때까지 대기
    if [ "$running" -ge "$parallel_count" ]; then
        wait
        running=0
    fi
done

# 남아있는 백그라운드 프로세스가 있다면 대기
wait

echo "모든 실험 완료. summary를 생성합니다."

# 모든 실험이 완료된 후 summary.py 실행
python summary.py --scenario_fol "$scenario_fol"

echo "실험 및 summary 생성 완료."
