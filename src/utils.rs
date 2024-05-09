use autometrics::autometrics;
use rayon::prelude::*;

#[autometrics]
pub fn convert_number_to_target_vec(num: usize) -> Vec<f64> {
    let mut v = vec![0.0; 10];
    if num < 10 {
        v[num] = 1.0;
    }
    v
}

#[autometrics]
fn find_max_index(v: Vec<f64>) -> Option<usize> {
    v.par_iter()
        .enumerate()
        .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
}

#[autometrics]
pub fn convert_result_vec_to_number(result_vec: Vec<f64>) -> usize {
    match find_max_index(result_vec) {
        Some(index) => index,
        None => panic!("Invalid inputs length"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_number_to_target_vec_valid() {
        let result = convert_number_to_target_vec(3);
        let expected = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_convert_number_to_target_vec_out_of_bounds() {
        let result = convert_number_to_target_vec(10);
        let expected = vec![0.0; 10];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_find_max_index_non_empty() {
        let vec = vec![0.2, 0.9, 0.7];
        let max_index = find_max_index(vec);
        assert_eq!(max_index, Some(1));
    }

    #[test]
    fn test_find_max_index_empty() {
        let vec: Vec<f64> = Vec::new();
        let max_index = find_max_index(vec);
        assert_eq!(max_index, None);
    }

    #[test]
    fn test_find_max_index_equal_elements() {
        let vec = vec![1.0, 1.0, 1.0];
        let max_index = find_max_index(vec);
        assert_eq!(max_index, Some(2));
    }

    #[test]
    fn test_convert_result_vec_to_number_valid() {
        let result_vec = vec![0.1, 0.5, 0.2, 0.2];
        let number = convert_result_vec_to_number(result_vec);
        assert_eq!(number, 1);
    }

    #[test]
    #[should_panic(expected = "Invalid inputs length")]
    fn test_convert_result_vec_to_number_invalid() {
        let result_vec = vec![];
        convert_result_vec_to_number(result_vec); // This should panic
    }
}
