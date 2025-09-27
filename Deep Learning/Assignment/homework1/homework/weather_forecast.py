from typing import Tuple
import torch


class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        """
        You are given a list of 10 weather measurements per day.
        Save the data as a PyTorch (num_days, 10) tensor,
        where the first dimension represents the day,
        and the second dimension represents the measurements.
        """
        self.data = torch.as_tensor(data_raw, dtype=torch.float32).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the max and min temperatures per day

        Returns:
            min_per_day: tensor of size (num_days,)
            max_per_day: tensor of size (num_days,)
        """
        min_per_day = torch.min(self.data, dim=1).values 
        max_per_day = torch.max(self.data, dim=1).values
        return min_per_day, max_per_day

    def find_the_largest_drop(self) -> torch.Tensor:
        """
        Find the largest change in day over day average temperature.
        This should be a negative number.

        Returns:
            tensor of a single value, the difference in temperature
        """
        avg_per_day = torch.mean(self.data, dim=1)  
        day_diff = avg_per_day[1:] - avg_per_day[:-1]  
        largest_drop = torch.min(day_diff)  
        return largest_drop  

    def find_the_most_extreme_day(self) -> torch.Tensor:
        """
        For each day, find the measurement that differs the most from the day's average temperature

        Returns:
            tensor with size (num_days,)
        """
        avg_per_day = torch.mean(self.data, dim=1, keepdim=True)  
        diffs = torch.abs(self.data - avg_per_day)  
        max_diff_indices = torch.argmax(diffs, dim=1)  
        
        # Gather the actual temperature values using advanced indexing
        num_days = self.data.shape[0]
        day_indices = torch.arange(num_days)
        extreme_temps = self.data[day_indices, max_diff_indices]  # shape (N,)
        
        return extreme_temps

    def max_last_k_days(self, k: int) -> torch.Tensor:
        """
        Find the maximum temperature over the last k days

        Returns:
            tensor of size (k,)
        """
        last_k_days = self.data[-k:]  
        max_per_day = torch.max(last_k_days, dim=1).values  # shape (k,)
        return max_per_day

    def predict_temperature(self, k: int) -> torch.Tensor:
        """
        From the dataset, predict the temperature of the next day.
        The prediction will be the average of the temperatures over the past k days.

        Args:
            k: int, number of days to consider

        Returns:
            tensor of a single value, the predicted temperature
        """
        last_k = self.data[-k:]  
        predicted_temp = torch.mean(last_k) 
        return predicted_temp 

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        """
        Given a list of 10 temperature measurements, find the day in the dataset
        that most closely matches the given measurements.

        We use sum of absolute differences per measurement.

        Args:
            t: tensor of size (10,), temperature measurements

        Returns:
            tensor of a single value, the index of the closest data element
        """
        diffs = torch.sum(torch.abs(self.data - t), dim=1)  
        idx = torch.argmin(diffs)  
        return idx.long()
