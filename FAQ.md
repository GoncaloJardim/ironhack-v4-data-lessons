# Questions gathered from students through the cohorts.

## PANDAS

### Steps to take when working with a new dataset  
1. Check dataset shape, nulls, clean if needed  
2. Come up with questions your interested in, go from high-level to low-level    
EXAMPLES TO BE ADDED
3. Start EDA, find issues with the dataset and fix them as you go. The issues can be things like:
    * wrong data types
    * wrong values, e.g. ages around 200 years
    * inconsistent values, e.g. M, male, men for the same gender
4. Repeat 2. and 3. until you've answered your most pressing questions

### When to use a quick EDA vs a more formal analysis
* **quick EDA**: Good for simple questions or questions that don't require 100% accuracy or for when you're in a time crunch. Orfor when you want to find ideas.
* **formal analysis**: For when you have time to do it, mainly. It's typically when you need to answer the most important business questions, and you need to get it right.

### Standard workflow for cleaning messy data efficiently
The 2 and 3 from above. There's no cookie-cutter answer. At most, you can automate data validation using Pydantic (easy mode) or great_expectations (hard mode, not worth it for static data)

### Grouping Methods by category adding common parameters

## STATS

### Probability Distributions summary for different types of data and nature of underlying process

These probability distributions are good at modelling real-life events because their underlying assumptions closely follow what happens in reality.

* Discrete Data (Countable Outcomes)
    * Bernoulli: Single binary outcome (success/failure). This works if the probability of success/failure doesn't change as you try multiple times, i.e. the events are independent of each other.
    * Binomial: Number of successes in a fixed number of trials.
    * Poisson: Number of events occurring in a fixed time/space interval.
    * Geometric: Number of trials until the first success.
* Continuous Data (Uncountable, Real Numbers)
    * Uniform: All values in a range are equally likely.
    * Normal (Gaussian): Data clusters symmetrically around a mean.
    * Exponential: Models time until an event occurs (e.g., failure rate).
    * Gamma: Generalization of the exponential, useful for waiting times.
    * Beta: Models probabilities or proportions.
* Other Specialized Distributions
    * Log-Normal: Used when data is skewed and follows a multiplicative process.
    * Weibull: Common in reliability analysis for failure times.
    * Multinomial: Extension of binomial for multiple categorical outcomes.

### Is there a complementary function to cdf?  (sf(x) = 1 - cdf(x))

### Vanguard project

#### Proper way to manage Vanguard project

#### Discrepancies between results management

#### Experiment Evaluation adequate test (attaching my recommendation to Borja that morning where I seemed convinced about a Z-test—but you said it should have been a T-test instead)

#### Repos: Worth the time they take? Other ways to present our work (Miguel kindly show his last Sat)
Having a good repo is definitely worth it, be it on GitHub or Tableau Public - this depends on whether you want to focus more on data science (Git) or data analysis (Tableau).

But the README.md doesn't have to be extra technical. 
* If you built something like a streamlit app, I would highly encourage you do a video to showcase how it works, presentation included, and put that at the very beginning. 
* If it's a model, give examples of how it performs on different data and how it's solving a problem. 
* If it's an analysis, focus on the impact and main findings 

Then, the custom is to give instructions on how to replicate your work, especially for Streamlit apps and ML models. Then, you can get into the main steps of what you did. But product/impact always come first.

## VIZ

### Best chart type for the different types of data recap
I think it's best to use [Data-to-Viz](https://www.data-to-viz.com/) for this!

### How can I ensure my visualizations are accessible and clear for stakeholders? How much knowledge do I assume they have?
* You HAVE to understand the message your visualization is sending very well. If that is unclear to you, it will also be unclear to other people. The message can follow the SETUP + CONFLICT + RESOLUTION template to make it more impactful.
* You simplify as much as you can without losing the essence of what you try to say. Less colors, less lines, less everything, until you have the essential.
* You use design principle, e.g. white space, visual ordering, pre-attentive features, to guide the audience towards the conclusion you want them to reach
* You repeat the main line with as many visual cues as possible. It's the advertising of data science. Think color coordination between a message and a statistic, combining words (up), emojis (⬆️), colors (🟩 / 🟦) with the trend 📈 you notice. 

### What challenges do analysts typically face when presenting results to non-technical stakeholders?  
* Understanding why their analysis is relevant to the business. You can tell an analyst doesn't understand the business by the data they choose to analyze and the things they focus on - there are many interesting things in data, but not all of them are useful / impactful.
* Simplifying the analysis. It's easy to get lost in the details and assume people know what you're talking about.

### Difference between statistical significance and practical significance
Practical significance is: **How wrong can I afford to be about this** It's a decision-making question that requires a decision-making framework. Things to keep in mind:
* What is the cost of getting it wrong? Do we lose $10, $100, $1,000, $1M? Will people die?
    * If I get a false positive, what's the worst that can happen? e.g. if I wrongly say that the new UI interface design isbetter, how many clients will I lose?
    * If I get a false negative, what's the worst that can happen? e.g. if I wrongly say that a treatment is not working, whatopportunity am I losing?
* How much would I be willing to pay to get this right? 
* How reversible is the decision I will make based on this result? A lot of decisions are easily reversible, i.e. you don'tneed to worry too much about a good significance level

Statistical significance is the number you put on the practical significance - both for the false positive cost (alpha) and false negative cost (beta).

## TABLEAU

### How can we prevent generating duplicates? Also, Alberto pointed to Power Query in Power BI looking for a similar option in Tableau (answer: issue in the original dataset or bad join—revert to pre-joining state to check numbers, explore and analyze in Python or use viz to count the numbers).

### Aggregation levels: your explanation about profit/sales vs SUM(profit)/SUM(sales) was very good, bt I needed to understand the implications of row-level vs aggregated calculations a bit better and settled on using the later unless specifically needed to calculate at a row level and analyze individual ratios before aggregation.

## MOST DAYS

### What mistakes should I watch out for when using this method/tool? (I generally try to prevent common pitfalls and improve efficiency)

### Use cases for this concept in real-world projects

### Recommended resources for diving deeper