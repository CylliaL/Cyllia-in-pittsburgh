create_monthly_prices <- function(daily_date,daily_price){
  n_obs_daily = length(daily_date) 
  year_date = format(daily_date,"%Y")
  month_date = format(daily_date,"%m")
  day_date = format(daily_date,"%d")
  start_date = as.Date(paste(year_date,month_date,1,sep="-"),"%Y-%m-%d")[1]#daily_date[1];
  end_date = daily_date[n_obs_daily]
  year_seq = year_date[1]: year_date[length(year_date)]
  month_seq = 1:12
  monthly_date = seq(from = start_date, to = end_date, by = "month")
  monthly_date = format(monthly_date,"%Y-%m")
  n_obs_monthly = length(monthly_date)
  monthly_price = rep(NA, n_obs_monthly)
  for (i1 in 1:length(year_seq)) {
    for (i2 in 1:length(month_seq)) {
      year_idx = year_seq[i1]
      month_idx = month_seq[i2]
      idx = year_date == year_idx & as.integer(month_date) == month_idx
      temp_price = daily_price[idx,]
      if (length(temp_price) != 0){
        monthly_price[(12*(i1-1)+i2)] = temp_price[length(temp_price)]
      }
    }
  }
  obj = list(monthly_date = monthly_date, monthly_price = monthly_price)
  return(obj)
}