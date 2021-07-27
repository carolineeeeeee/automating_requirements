script_dir=$(dirname $0)
for f in $(ls $script_dir/../data | grep -v README.md);
do
  to_remove="$script_dir/../data/$f"
  echo "Removing $to_remove"
  rm -rf $to_remove;
done