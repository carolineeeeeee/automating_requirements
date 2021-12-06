echo "remove dataset"
script_dir=$(dirname $0)
for f in $(ls $script_dir/../data | grep -v README.md | grep -v frost-images);
do
  to_remove="$script_dir/../data/$f"
  echo "Removing $to_remove"
  rm -rf $to_remove;
done

echo "remove bootstrap images directory"
rm -rf $script_dir/../bootstrap_data
rm -rf $script_dir/../bootstrap_data_c

echo "remove all jobs in queue and finished jobs"
rm -rf $script_dir/../jobs
rm -rf $script_dir/../finished_jobs

echo "remote data/imagenet"
if [ -e $script_dir/../data/imagenet ]; then rm -rf $script_dir/../data/imagenet; fi
