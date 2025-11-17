# Detransform Centered/Scaled Data

This function back transforms centered/scaled data.

## Usage

``` r
detransform(x_data, ...)
```

## Arguments

- x_data:

  Model matrix that has been centered and/or scaled.

- ...:

  Additional arguments specifying centering/scaling attributes.

## Value

Returns de-centered and de-scaled model matrix.

## Details

The following additional arguments can be passed:

- `attr_center` : Centering attributes.

  - If none specified, `attr(x_data,'scaled:center')` is utilized.

- `attr_scale` : Scaling attributes.

  - If none specified, `attr(x_data,'scaled:scale')` is utilized.

## See also

- [`scale`](https://rdrr.io/r/base/scale.html) for centering and
  scaling.

- [`all.equal`](https://rdrr.io/r/base/all.equal.html) for testing "near
  equality".

## Examples

``` r
# Set seed for reproducibility
set.seed(1964)
# Generate a 10x10 matrix with random numbers
original_data <- matrix(rnorm(100), nrow = 10)
# Scale and center the data
scaled_centered_data <- scale(original_data, center = TRUE,
scale = TRUE)
# Transform the scaled/centered data back to its original form
original_data_recovered <- detransform(scaled_centered_data)
# Compare the original data and the recovered data
all.equal(original_data,original_data_recovered)
#> [1] TRUE
```
