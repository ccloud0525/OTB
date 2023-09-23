

rm -rf   ../ts_benchmark/baselines/third_party/tods

REPO_URL='https://gitee.com/zhangbuang/tods.git'
# COMMIT_HASH=$2
# DEST_FOLDER=$3

# 克隆仓库到指定目录
git clone $REPO_URL

mv  ./tods/tods ../ts_benchmark/baselines/third_party
rm -rf ./tods