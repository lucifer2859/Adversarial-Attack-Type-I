for ((i=0;i<10;i++))
do
	python run_mnist.py --phase attack --test_index $i --target_label 8
done