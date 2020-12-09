library(tidyverse)
library(sampling)
options(digits = 4)

# read mass_shooting dataset and replace -999 to NA
df <- read_csv('mass_shooting.csv', na = c(-999, "NA") )
glimpse(df)
# a problem is column names are not well-formated, so I had to clean it up 
# and sort the data by date.
colnames(df)
# fix Date

df <- df %>% rename(
  Incideent_are = `Incident Area`,
  Location_type = `Open/Close Location`,
  Shooter_status = `Shooter status`,
  Shooter_number = `No. of shooter/suspect`,
  Total_vicitimes = `Total victims`,
  Policeman_killed = `Policeman Killed`,
  Suspect_age = Age,
  Employeed = `Employeed (Y/N)`,
  Employed_at = `Employed at`,
  Mental_problem = `Mental Health Issues`
) %>% select(4, 5, 3, 7:24, -10) %>% mutate(
  Date = as.Date(df$Date, "%m/%d/%y"),
  Mental_problem = ifelse(Mental_problem == 'yes', 'Yes', Mental_problem),
  Mental_problem = replace_na(Mental_problem, 'Unknown'),
  Cause = replace_na(Cause, 'Unknown'),
  Target = replace_na(Target, 'Unknown')
) %>% arrange(desc(Date) ) %>% filter(Date < '2020-12-31')
glimpse(df)

unique(df$Mental_problem)
# analysis categorical variables of Target type and Cause

table(df$Target)
barplot(table(df$Target), las = 2, cex.names = 0.8, cex.axis = 0.8, 
        col = 'black', main = 'Mass shooting cases count by target types',
        ylab = 'Count')

barplot(table(df$Cause), las = 2, cex.names = 0.8, cex.axis = 0.8, 
        col = 'lightblue', main = 'Mass shooting cases count by causes',
        ylab = 'Count')

# analysis numeric variable of Fatality
hist(df$Fatalities)
ggplot(df) + geom_histogram(aes(x = Fatalities, y = ..density..),bins = 20, 
color="darkblue", fill="lightblue") + labs(title = 'Fatalities distribution')



# Check if total victim number and fatalities are related to cause or suspect mental problem

# first to check correlation between deaths and victims, they should strongly positive associated
cor(df$Total_vicitimes, df$Fatalities)

df %>% group_by(Mental_problem) %>% summarise(
  death = mean(Fatalities),
  victim = mean(Total_vicitimes)
) %>% arrange( desc(death, victim))

ggplot(df) + geom_histogram(aes(Fatalities, y = ..density.., 
                                fill = Mental_problem))


df %>% group_by(Cause) %>% summarise(
  death = mean(Fatalities),
  victim = mean(Total_vicitimes)
) %>% arrange( desc(death, victim))


# see the distribution of total victims
prop.table(table(df$Total_vicitimes))
plot(prop.table(table(df$Total_vicitimes)), type = 'h', 
     ylab = 'probability', xlab = 'total victims', main = 'PDF of total victims')

# remove the outlier 585 and plot the pdf again

victim2 = subset(df$Total_vicitimes, df$Total_vicitimes != max(df$Total_vicitimes))
plot(prop.table(table(victim2)), type = 'h', 
     ylab = 'probability', xlab = 'total victims', main = 'PDF of total victims')





# Sampling 

# simple random sampling
victim = df$Total_vicitimes

N = nrow(df)
sample_size = 50
n_samples = 30
xbar = numeric(n_samples)
for (i in 1:n_samples) {
  p = sample(victim, sample_size)
  xbar[i] = mean(p)
}
xbar
mean(xbar)
mean(victim)



# bootstrapping 

# total number of samples from the population with replacement
bootstrapping = sample(victim, N, replace = T)
par(mfrow = c(1, 2))
plot(prop.table(table(bootstrapping)), type = 'l', 
     col = 'darkgreen', main = 'PDF of victimes numbers 
     by sampling with replacement',
     ylab = 'probability', xlab = 'victimes', xlim = c(0, 100))
plot(prop.table(table(df$Total_vicitimes)), type = 'l', 
     col = 'darkgreen', main = 'PDF of victimes numbers',
     ylab = 'probability', xlab = 'victimes', xlim = c(0, 100))


# systematic sampling with step size 20
par(mfrow = c(1, 2))
k = ceiling(N / 20)
sys_samples = df[seq(from = sample(k, 1), by = k, to= N), ]
plot(prop.table(table(sys_samples$Total_vicitimes)), type = 'l', 
     col = 'darkgreen', main = 'PDF of victimes numbers by 
     systematic sampling',
     ylab = 'probability', xlab = 'victimes')

plot(prop.table(table(df$Total_vicitimes)), type = 'l', 
     col = 'darkgreen', main = 'PDF of victimes numbers',
     ylab = 'probability', xlab = 'victimes', xlim = c(0, 100))

par(mfrow = c(1, 1))


# inclusion probability
pik <- inclusionprobabilities(df$Fatalities, 100)
sp <- UPsystematic(pik)
in_samples = df[sp != 0, ]
par(mfrow = c(1, 2))
plot(prop.table(table(in_samples$Total_vicitimes)), type = 'l', 
     col = 'darkgreen', main = 'PDF of victimes numbers 
     by inclusion probability',
     ylab = 'probability', xlab = 'victimes', ylim = c(0, 0.3), 
     xlim = c(0, 100))

plot(prop.table(table(df$Total_vicitimes)), type = 'l', 
     col = 'darkgreen', main = 'PDF of victimes numbers',
     ylab = 'probability', xlab = 'victimes',
     ylim = c(0, 0.3), xlim = c(0, 100))
par(mfrow = c(1, 1))



# stratified sampling
df_stra <- df[, c('Mental_problem', 'Total_vicitimes')]
freq <- table(df_stra$Mental_problem)
st_size = 100 * freq / sum(freq)
stra_reg <- strata(df_stra, stratanames = 'Mental_problem', size = st_size, method = 'srswor')
stra_simples = getdata(df_stra, stra_reg)$Total_vicitimes
par(mfrow = c(1, 2))
plot(prop.table(table(stra_simples)), type = 'l', 
     col = 'darkgreen', main = 'PDF of victimes numbers 
     by stratified sampling',
     ylab = 'probability', xlab = 'victimes', ylim = c(0, 0.3), 
     xlim = c(0, 100))

plot(prop.table(table(df$Total_vicitimes)), type = 'l', 
     col = 'darkgreen', main = 'PDF of victimes numbers',
     ylab = 'probability', xlab = 'victimes',
     ylim = c(0, 0.3), xlim = c(0, 100))
par(mfrow = c(1, 1))








