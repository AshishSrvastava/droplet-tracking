def properties_from_volume_fraction(volume_fraction_glyc, T):
    volume_fraction_water = 1 - volume_fraction_glyc

    density_glycerol = 1273.3 - 0.6121 * T  # kg/m^3
    density_water = 1000 * (1 - ((abs(T - 3.98)) / 615) ** 1.71)


def main():
    print("Hello, world!")


if __name__ == "__main__":
    main()
