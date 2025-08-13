$t_values = @(2, 3)

$eps_values = @(1.0, 2.0, 3.0, 4.0, 5.0)

# 각 t 값에 대하여 반복
foreach ($t in $t_values) {
    # 각 eps 값에 대하여 반복
    foreach ($eps in $eps_values) {
        $command = "python pm_transform_train_test_batch_label.py --t $t --eps $eps --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/credit.csv --label_col label"

        Write-Host "Executing: $command" -ForegroundColor Green

        Invoke-Expression $command
    }
}

# 각 t 값에 대하여 반복
foreach ($t in $t_values) {
    # 각 eps 값에 대하여 반복
    foreach ($eps in $eps_values) {
        $command = "python pm_transform_train_test_batch_label.py --t $t --eps $eps --transform_label_numerical True --transform_label_categorical False --transform_label_log True --csv_path data/OnlineNewsPopularity.csv --label_col shares"

        Write-Host "Executing: $command" -ForegroundColor Green

        Invoke-Expression $command
    }
}

# 각 t 값에 대하여 반복
foreach ($t in $t_values) {
    # 각 eps 값에 대하여 반복
    foreach ($eps in $eps_values) {
        $command = "python pm_transform_train_test_batch_label.py --t $t --eps $eps --transform_label_numerical True --transform_label_categorical False --transform_label_log False --csv_path data/CASP.csv --label_col RMSD"

        Write-Host "Executing: $command" -ForegroundColor Green

        Invoke-Expression $command
    }
}

foreach ($t in $t_values) {
    # 각 eps 값에 대하여 반복
    foreach ($eps in $eps_values) {
        $command = "python pm_transform_train_test_batch_label.py --t $t --eps $eps --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/gamma.csv --label_col label"

        Write-Host "Executing: $command" -ForegroundColor Green

        Invoke-Expression $command
    }
}

foreach ($t in $t_values) {
    # 각 eps 값에 대하여 반복
    foreach ($eps in $eps_values) {
        $command = "python pm_transform_train_test_batch_label.py --t $t --eps $eps --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/shuttle.csv --label_col label"

        Write-Host "Executing: $command" -ForegroundColor Green

        Invoke-Expression $command
    }
}

foreach ($t in $t_values) {
    # 각 eps 값에 대하여 반복
    foreach ($eps in $eps_values) {
        $command = "python pm_transform_train_test_batch_label.py --t $t --eps $eps --transform_label_numerical False --transform_label_categorical True --transform_label_log False --csv_path data/wine.csv --label_col label"

        Write-Host "Executing: $command" -ForegroundColor Green

        Invoke-Expression $command
    }
}

Write-Host "모든 작업이 완료되었습니다." -ForegroundColor Cyan