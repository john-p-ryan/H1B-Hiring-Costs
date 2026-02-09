clear all
set more off
do globals

foreach yyyy of numlist 2021/2023 {

    import delimited "${data}/TRK_13139_FY`yyyy'.csv", clear varn(1)
    drop if lottery_year != "`yyyy'"

    * Clean up feins. They are supposed to be 9 digits; calculate odds among only those that have an fein
    order fein
    gen fein_len = strlen(fein)
    drop if fein_len != 9
    drop fein_len
    
    sort fein
    gen counter = 1
    gen success = status_type == "SELECTED"
    collapse (sum) registrations = counter success, by(fein)
    gen success_rate = success / registrations

    * Drop outlying firms
    qui summ registrations, d
    drop if registrations > r(p99)


    frame copy default agg_rate, replace
    frame agg_rate {
        collapse (sum) registrations success
        gen success_rate_agg = success / registrations
        loc agg_success_rate = success_rate_agg[1]
    }
    frame drop agg_rate


    * Success rate distribution - going to combine these into one graph later, hence the 'nodraw' optionh
    tw histogram success_rate, color(none) fcolor(ebblue%70) xline(`agg_success_rate', lc(orange)) name(success_rate_`yyyy') ///
    xlab(, nogrid) ylab(, nogrid format(%9.0fc)) xtitle("Ex-Post Success Rate") freq title("Fiscal Year `yyyy'") nodraw

    * Calculate conditional variances by quartile of registration dbn
    foreach tile of numlist 33 67 {
        egen registrations_p`tile' = pctile(registrations), p(`tile')
    }
    
    gen tile = 1       if registrations <= registrations_p33
    replace  tile = 2  if registrations > registrations_p33 & registrations <= registrations_p67
    replace  tile = 3  if registrations > registrations_p67 & !mi(registrations)
    collapse (sd) success_rate_std = success_rate, by(tile)
    gen var = success_rate_std^2

    * Plot variance by registration tercile
    la def tercile 1 "First Tercile" 2 "Second Tercile" 3 "Third Tercile"
    la val tile tercile
    graph bar var, over(tile) ytitle("Success Rate Variance") ylabel(,nogrid) name("cond_var_`yyyy'") title("Fiscal Year `yyyy'") ///
    nodraw bar(1, color(ebblue)) bar(2, color(ebblue)) bar(3, color(ebblue))

    
}

* Create a combined success rate graph
graph combine success_rate_2021 success_rate_2022 success_rate_2023, cols(2) rows(2) name("success_rate_by_year")
graph export "${graphs}/success_rate_by_year.pdf", name("success_rate_by_year") replace

* Create a combined conditional success rate variance graph
graph combine cond_var_2021 cond_var_2022 cond_var_2023, cols(2) rows(2) name("cond_var_by_year")
graph export "${graphs}/cond_var_by_year.pdf", name("cond_var_by_year") replace


