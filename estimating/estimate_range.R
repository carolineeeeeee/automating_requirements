library(np ) ; library( car ) ; library( stats )
library( graphics )
library( SemiPar ) 
library(ggplot2)

cifar10_tranformations <- c('brightness', 'contrast', 'frost', 'jpeg_compression')
orig_acc = 0.967
print('cifar10')
for (transformation in cifar10_tranformations) {
	print(transformation)
	print('correctness-preservation')
	csv_filename = paste('/Users/caroline/Desktop/REforML/git_repos/automating_requirements/estimating/cifar10/','cifar10_c_p_', transformation, '.csv', sep = "")
	all_point <- read.table(csv_filename, colClasses=c("numeric", "numeric", "numeric", "numeric"), header = TRUE, numerals = "no.loss", sep=',')
	all_point.IQA <- as.vector(all_point[,'IQA'])
	all_point.Counts<-as.vector(all_point[,"Count"])
	all_point.Accuracy <- as.vector(all_point[,"Accuracy"])
	all_point.Num_Acc <- as.vector(all_point[,"Num_Accs"])
	
	all_fit =smooth.spline( all_point.IQA , all_point.Accuracy, w=all_point.Counts/min(all_point.Counts), cv=T  )
	res <- (all_fit$yin - all_fit$y)/(1-all_fit$lev)      # jackknife residuals
	sigma <- sqrt(var(res))                     # estimate sd
	#all_point.upper <- all_fit$y + 2.0*sigma*sqrt(all_fit$lev)   # upper 95% conf. band
	#all_point.lower <- all_fit$y - 2.0*sigma*sqrt(all_fit$lev)   # lower 95% conf. band
	all_point.upper <- all_fit$y + 1.39*sigma*sqrt(all_fit$lev)   # upper 83% conf. band
	all_point.lower <- all_fit$y - 1.39*sigma*sqrt(all_fit$lev)   # lower 83% conf. band
	matplot(all_fit$x, cbind(all_point.upper, all_fit$y, all_point.lower), type="plp", pch=21:23, xlim=c( 0, 1) , ylim=c( 0, 1 ))
	
	p_values <- c()
	for (i in 1:length(all_point.IQA)){
		binom_result <- binom.test(round(all_point.Counts[i] * predict(all_fit, all_point.IQA[i])$y, digits=0), all_point.Counts[i], orig_acc)
		p_values <- c(p_values, binom_result$p.value)
	}

	i <- 1
	while (i <= length(p_values)){
		if (p_values[i] < 0.05){
			break
		}
		i = i+1
	}
	if (i > length(p_values)){
		print(all_point.IQA[length(p_values)])
	} else {
		print(all_point.IQA[i])
	}
	
	print('prediction-preservation')	
	csv_filename = paste('/Users/caroline/Desktop/REforML/git_repos/automating_requirements/estimating/cifar10/','cifar10_p_p_', transformation, '.csv', sep = "")
	all_point <- read.table(csv_filename, colClasses=c("numeric", "numeric", "numeric", "numeric"), header = TRUE, numerals = "no.loss", sep=',')
	all_point.IQA <- as.vector(all_point[,'IQA'])
	all_point.Counts<-as.vector(all_point[,"Count"])
	all_point.Accuracy <- as.vector(all_point[,"Preserved"])
	all_point.Num_Acc <- as.vector(all_point[,"Num_Pre"])
	
	all_fit =smooth.spline( all_point.IQA , all_point.Accuracy, w=all_point.Counts/min(all_point.Counts), cv=T  )
	res <- (all_fit$yin - all_fit$y)/(1-all_fit$lev)      # jackknife residuals
	sigma <- sqrt(var(res))                     # estimate sd
	#all_point.upper <- all_fit$y + 2.0*sigma*sqrt(all_fit$lev)   # upper 95% conf. band
	#all_point.lower <- all_fit$y - 2.0*sigma*sqrt(all_fit$lev)   # lower 95% conf. band
	all_point.upper <- all_fit$y + 1.39*sigma*sqrt(all_fit$lev)   # upper 83% conf. band
	all_point.lower <- all_fit$y - 1.39*sigma*sqrt(all_fit$lev)   # lower 83% conf. band
	matplot(all_fit$x, cbind(all_point.upper, all_fit$y, all_point.lower), type="plp", pch=21:23, xlim=c( 0, 1) , ylim=c( 0, 1 ))
	
	p_values <- c()
	left = ceiling(sum(all_point.Counts) * 0.5)
	total_count = 0
	total_preserved = 0
	for (i in 1:length(all_point.Counts)){
		left = left - all_point.Counts[i]
		total_count = total_count + all_point.Counts[i]
		total_preserved = total_preserved + all_point.Num_Acc[i]
		if (left <= 0){
			break
		}
	}
	
	pre_acc <- total_preserved/total_count
	for (i in 1:length(all_point.IQA)){
		binom_result <- binom.test(round(all_point.Counts[i] * predict(all_fit, all_point.IQA[i])$y, digits=0), all_point.Counts[i], pre_acc)
		p_values <- c(p_values, binom_result$p.value)
	}
	i <- 1
	while (i <= length(p_values)){
		if (p_values[i] < 0.05){
			break
		}
		i = i+1
	}
	if (i > length(p_values)){
		print(all_point.IQA[length(p_values)])
	} else {
		print(all_point.IQA[i])
	}
}

print('imagenet')
imagenet_transformations <- c('RGB', 'brightness', 'contrast', 'gaussian_noise', 'defocus_blur', 'frost', 'jpeg_compression', 'color_jitter')
orig_acc = 0.970

for (transformation in imagenet_transformations) {
	print(transformation)
	print('correctness-preservation')
	csv_filename = paste('/Users/caroline/Desktop/REforML/git_repos/automating_requirements/estimating/imagenet/','imagenet_c_p_', transformation, '.csv', sep = "")
	all_point <- read.table(csv_filename, colClasses=c("numeric", "numeric", "numeric", "numeric"), header = TRUE, numerals = "no.loss", sep=',')
	all_point.IQA <- as.vector(all_point[,'IQA'])
	all_point.Counts<-as.vector(all_point[,"Count"])
	all_point.Accuracy <- as.vector(all_point[,"Accuracy"])
	all_point.Num_Acc <- as.vector(all_point[,"Num_Accs"])
	
	all_fit =smooth.spline( all_point.IQA , all_point.Accuracy, cv=T)
	res <- (all_fit$yin - all_fit$y)/(1-all_fit$lev)      # jackknife residuals
	sigma <- sqrt(var(res))                     # estimate sd
	#all_point.upper <- all_fit$y + 2.0*sigma*sqrt(all_fit$lev)   # upper 95% conf. band
	#all_point.lower <- all_fit$y - 2.0*sigma*sqrt(all_fit$lev)   # lower 95% conf. band
	all_point.upper <- all_fit$y + 1.39*sigma*sqrt(all_fit$lev)   # upper 83% conf. band
	all_point.lower <- all_fit$y - 1.39*sigma*sqrt(all_fit$lev)   # lower 83% conf. band
	matplot(all_fit$x, cbind(all_point.upper, all_fit$y, all_point.lower), type="plp", pch=21:23, xlim=c( 0, 1) , ylim=c( 0, 1 ))
	
	p_values <- c()
	for (i in 1:length(all_point.IQA)){
		binom_result <- binom.test(round(all_point.Counts[i] * predict(all_fit, all_point.IQA[i])$y, digits=0), all_point.Counts[i], orig_acc)
		p_values <- c(p_values, binom_result$p.value)
	}
	#print(p_values)
	i <- 1
	while (i <= length(p_values)){
		#print(p_values[i])
		if (p_values[i] < 0.05){
			break
		}
		i = i+1
	}
	if (i > length(p_values)){
		print(all_point.IQA[length(p_values)])
	} else {
		print(all_point.IQA[i])
	}
	
	print('prediction-preservation')	
	csv_filename = paste('/Users/caroline/Desktop/REforML/git_repos/automating_requirements/estimating/imagenet/','imagenet_p_p_', transformation, '.csv', sep = "")
	all_point <- read.table(csv_filename, colClasses=c("numeric", "numeric", "numeric", "numeric"), header = TRUE, numerals = "no.loss", sep=',')
	all_point.IQA <- as.vector(all_point[,'IQA'])
	all_point.Counts<-as.vector(all_point[,"Count"])
	all_point.Accuracy <- as.vector(all_point[,"Preserved"])
	all_point.Num_Acc <- as.vector(all_point[,"Num_Pre"])

	all_fit =smooth.spline( all_point.IQA , all_point.Accuracy, cv=T)
	res <- (all_fit$yin - all_fit$y)/(1-all_fit$lev)      # jackknife residuals
	sigma <- sqrt(var(res))                     # estimate sd
	#all_point.upper <- all_fit$y + 2.0*sigma*sqrt(all_fit$lev)   # upper 95% conf. band
	#all_point.lower <- all_fit$y - 2.0*sigma*sqrt(all_fit$lev)   # lower 95% conf. band
	all_point.upper <- all_fit$y + 1.39*sigma*sqrt(all_fit$lev)   # upper 83% conf. band
	all_point.lower <- all_fit$y - 1.39*sigma*sqrt(all_fit$lev)   # lower 83% conf. band
	matplot(all_fit$x, cbind(all_point.upper, all_fit$y, all_point.lower), type="plp", pch=21:23, xlim=c( 0, 1) , ylim=c( 0, 1 ))
	
	p_values <- c()
	left = ceiling(sum(all_point.Counts) * 0.5)
	total_count = 0
	total_preserved = 0
	for (i in 1:length(all_point.Counts)){
		left = left - all_point.Counts[i]
		total_count = total_count + all_point.Counts[i]
		total_preserved = total_preserved + all_point.Num_Acc[i]
		if (left <= 0){
			break
		}
	}
	
	pre_acc <- total_preserved/total_count
	for (i in 1:length(all_point.IQA)){
		binom_result <- binom.test(round(all_point.Counts[i] * predict(all_fit, all_point.IQA[i])$y, digits=0), all_point.Counts[i], pre_acc)
		p_values <- c(p_values, binom_result$p.value)
	}

	i <- 1
	while (i <= length(p_values)){
		#print(p_values[i])
		if (p_values[i] < 0.05){
			break
		}
		i = i+1
	}
	if (i > length(p_values)){
		print(all_point.IQA[length(p_values)])
	} else {
		print(all_point.IQA[i])
	}
	
}


print('comparing imagenet subset results')
imagenet_transformations <- c('RGB', 'brightness', 'contrast', 'gaussian_noise', 'defocus_blur', 'frost', 'jpeg_compression', 'color_jitter') 

for (transformation in imagenet_transformations) {
	print(transformation)
	#########################first subset###########################
	csv_filename = paste('/Users/caroline/Desktop/REforML/git_repos/automating_requirements/estimating/sixty_percent/sixty_percent_1/','imagenet_c_p_', transformation, '.csv', sep = "")
	acc_all_point_1 <- read.table(csv_filename, colClasses=c("numeric", "numeric", "numeric", "numeric"), header = TRUE, numerals = "no.loss", sep=',')
	acc_all_point_1.IQA <- as.vector(acc_all_point_1[,'IQA'])
	acc_all_point_1.Counts<-as.vector(acc_all_point_1[,"Count"])
	acc_all_point_1.Accuracy <- as.vector(acc_all_point_1[,"Accuracy"])
	acc_all_point_1.Num_Acc <- as.vector(acc_all_point_1[,"Num_Accs"])
	
	acc_all_fit_1 =smooth.spline( acc_all_point_1.IQA , acc_all_point_1.Accuracy, cv=T)
	res <- (acc_all_fit_1$yin - acc_all_fit_1$y)/(1-acc_all_fit_1$lev)      # jackknife residuals
	sigma <- sqrt(var(res))                     # estimate sd
	#acc_all_point_1.upper <- acc_all_fit_1$y + 2.0*sigma*sqrt(acc_all_fit_1$lev)   # upper 95% conf. band
	#acc_all_point_1.lower <- acc_all_fit_1$y - 2.0*sigma*sqrt(acc_all_fit_1$lev)   # lower 95% conf. band
	acc_all_point_1.upper <- acc_all_fit_1$y + 1.39*sigma*sqrt(acc_all_fit_1$lev)   # upper 83% conf. band
	acc_all_point_1.lower <- acc_all_fit_1$y - 1.39*sigma*sqrt(acc_all_fit_1$lev)   # lower 83% conf. band
	
	csv_filename = paste('/Users/caroline/Desktop/REforML/git_repos/automating_requirements/estimating/sixty_percent/sixty_percent_1/','imagenet_p_p_', transformation, '.csv', sep = "")
	pre_all_point_1 <- read.table(csv_filename, colClasses=c("numeric", "numeric", "numeric", "numeric"), header = TRUE, numerals = "no.loss", sep=',')
	pre_all_point_1.IQA <- as.vector(pre_all_point_1[,'IQA'])
	pre_all_point_1.Counts<-as.vector(pre_all_point_1[,"Count"])
	pre_all_point_1.Accuracy <- as.vector(pre_all_point_1[,"Preserved"])
	pre_all_point_1.Num_Acc <- as.vector(pre_all_point_1[,"Num_Pre"])
	
	pre_all_fit_1 =smooth.spline( pre_all_point_1.IQA , pre_all_point_1.Accuracy, cv=T)
	res <- (pre_all_fit_1$yin - pre_all_fit_1$y)/(1-pre_all_fit_1$lev)      
	sigma <- sqrt(var(res))                     # estimate sd
	#pre_all_point_1.upper <- all_fit$y + 2.0*sigma*sqrt(all_fit$lev)   # upper 95% conf. band
	#pre_all_point_1.lower <- all_fit$y - 2.0*sigma*sqrt(all_fit$lev)   # lower 95% conf. band
	pre_all_point_1.upper <- pre_all_fit_1$y + 1.39*sigma*sqrt(pre_all_fit_1$lev)   # upper 83% conf. band
	pre_all_point_1.lower <- pre_all_fit_1$y - 1.39*sigma*sqrt(pre_all_fit_1$lev)   # lower 83% conf. band
	
	
	#########################second subset###########################
	csv_filename = paste('/Users/caroline/Desktop/REforML/git_repos/automating_requirements/estimating/sixty_percent/sixty_percent_2/','imagenet_c_p_', transformation, '.csv', sep = "")
	acc_all_point_2 <- read.table(csv_filename, colClasses=c("numeric", "numeric", "numeric", "numeric"), header = TRUE, numerals = "no.loss", sep=',')
	acc_all_point_2.IQA <- as.vector(acc_all_point_2[,'IQA'])
	acc_all_point_2.Counts<-as.vector(acc_all_point_2[,"Count"])
	acc_all_point_2.Accuracy <- as.vector(acc_all_point_2[,"Accuracy"])
	acc_all_point_2.Num_Acc <- as.vector(acc_all_point_2[,"Num_Accs"])
	
	acc_all_fit_2 =smooth.spline( acc_all_point_2.IQA , acc_all_point_2.Accuracy, cv=T)
	res <- (acc_all_fit_2$yin - acc_all_fit_2$y)/(1-acc_all_fit_2$lev)   
	sigma <- sqrt(var(res))                     # estimate sd
	#acc_all_point_2.upper <- acc_all_fit_2$y + 2.0*sigma*sqrt(acc_all_fit_2$lev)   # upper 95% conf. band
	#acc_all_point_2.lower <- acc_all_fit_2$y - 2.0*sigma*sqrt(acc_all_fit_2$lev)   # lower 95% conf. band
	acc_all_point_2.upper <- acc_all_fit_2$y + 1.39*sigma*sqrt(acc_all_fit_2$lev)   # upper 83% conf. band
	acc_all_point_2.lower <- acc_all_fit_2$y - 1.39*sigma*sqrt(acc_all_fit_2$lev)   # lower 83% conf. band

	csv_filename = paste('/Users/caroline/Desktop/REforML/git_repos/automating_requirements/estimating/sixty_percent/sixty_percent_2/','imagenet_p_p_', transformation, '.csv', sep = "")
	pre_all_point_2 <- read.table(csv_filename, colClasses=c("numeric", "numeric", "numeric", "numeric"), header = TRUE, numerals = "no.loss", sep=',')
	pre_all_point_2.IQA <- as.vector(pre_all_point_2[,'IQA'])
	pre_all_point_2.Counts<-as.vector(pre_all_point_2[,"Count"])
	pre_all_point_2.Accuracy <- as.vector(pre_all_point_2[,"Preserved"])
	pre_all_point_2.Num_Acc <- as.vector(pre_all_point_2[,"Num_Pre"])
	
	
	pre_all_fit_2 =smooth.spline( pre_all_point_2.IQA , pre_all_point_2.Accuracy, cv=T)
	res <- (pre_all_fit_2$yin - pre_all_fit_2$y)/(1-pre_all_fit_2$lev)      # jackknife residuals
	sigma <- sqrt(var(res))                     # estimate sd
	#pre_all_point_2.upper <- all_fit$y + 2.0*sigma*sqrt(all_fit$lev)   # upper 95% conf. band
	#pre_all_point_2.lower <- all_fit$y - 2.0*sigma*sqrt(all_fit$lev)   # lower 95% conf. band
	pre_all_point_2.upper <- pre_all_fit_2$y + 1.39*sigma*sqrt(pre_all_fit_2$lev)   # upper 83% conf. band
	pre_all_point_2.lower <- pre_all_fit_2$y - 1.39*sigma*sqrt(pre_all_fit_2$lev)   # lower 83% conf. band

	par(mfrow = c(2, 1)) 
	matplot(pre_all_fit_1$x, cbind(pre_all_point_1.upper, pre_all_fit_1$y, pre_all_point_1.lower), type="plp", pch=21:23, xlim=c( 0, 1) , ylim=c( 0, 1 ))
	matplot(pre_all_fit_2$x, cbind(pre_all_point_2.upper, pre_all_fit_2$y, pre_all_point_2.lower), type="plp", pch=21:23, xlim=c( 0, 1) , ylim=c( 0, 1 ))
	
	
	acc_comparison_result <- c()
	acc_IQA <- c()
	for (i in 1:length(acc_all_point_1.IQA)){
		for (j in 1:length(acc_all_point_2.IQA)){
			if (acc_all_point_1.IQA[i] == acc_all_point_2.IQA[j]){
				acc_IQA <- c(acc_IQA, acc_all_point_2.IQA[i])
				intersection = min(acc_all_point_1.upper[i], acc_all_point_2.upper[j]) - max(acc_all_point_1.lower[i], acc_all_point_2.lower[j])
				acc_comparison_result <- c(acc_comparison_result, intersection)
			}
		}
	}

	if (all(acc_comparison_result>0)){
		print("simlar plots")
	} else {
		print(acc_comparison_result)
		print("the difference is statistically significant")
	}
	
	pre_comparison_result <- c()
	pre_IQA <- c()
	
	for (i in 1:length(pre_all_point_1.IQA)){
		for (j in 1:length(pre_all_point_2.IQA)){
			if (pre_all_point_1.IQA[i] == pre_all_point_2.IQA[j]){
				pre_IQA <- c(pre_IQA, pre_all_point_2.IQA[i])
				intersection = min(pre_all_point_1.upper[i], pre_all_point_2.upper[j]) - max(pre_all_point_1.lower[i], pre_all_point_2.lower[j])
				pre_comparison_result <- c(pre_comparison_result, intersection)
			}
		}
	}

	if (all(pre_comparison_result>0)){
		print("simlar plots")
	} else {
		print(pre_comparison_result)
		print("the difference is statistically significant")
	}

}

