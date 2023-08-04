pub fn convert_number_to_target_vec(num: usize) -> Vec<f64> {
    let mut v = vec![0.0; 10];
    if num < 10 {
        v[num] = 1.0;
    }
    v
}

fn find_max_index(v: Vec<f64>) -> Option<usize> {
    v.iter()
        .enumerate()
        .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
}

pub fn convert_result_vec_to_number(result_vec: Vec<f64>) -> usize {
    match find_max_index(result_vec) {
        Some(index) => index,
        None => panic!("Invalid inputs length"),
    }
}
