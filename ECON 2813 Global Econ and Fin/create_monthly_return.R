create_monthly_return <- function(daily_date,daily_return){
  n_obs_daily = length(daily_date) 
  year_date = format(daily_date,"%Y")
  month_date = format(daily_date,"%m")
  day_date = format(daily_date,"%d")
  start_date = daily_date[1];
  end_date = daily_date[n_obs_daily]
  year_seq = year_date[1]: year_date[length(year_date)]
  month_seq = 1:12
  monthly_date = seq(from = start_date, to = end_date, by = "month")
  monthly_date = format(monthly_date,"%Y-%m")
  n_obs_monthly = length(monthly_date)
  monthly_return = rep(NA, n_obs_monthly)
  monthly_log_return = rep(NA, n_obs_monthly)
  for (i1 in 1:length(year_seq)) {
    for (i2 in 1:length(month_seq)) {
      year_idx = year_seq[i1]
      month_idx = month_seq[i2]
      idx = year_date == year_idx & as.integer(month_date) == month_idx
      temp_return = daily_return[idx,]
      if (length(temp_return) != 0){
        monthly_return[(12*(i1-1)+i2)] = prod(1+temp_return)-1
        monthly_log_return[(12*(i1-1)+i2)] = sum(log(1+temp_return))
      }
    }
  }
  obj = list(monthly_date = monthly_date, monthly_return = monthly_return, monthly_log_return = monthly_log_return)
  return(obj)
}