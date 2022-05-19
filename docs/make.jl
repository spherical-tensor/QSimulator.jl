using Documenter, QSimulator


enabled_pages = [
    "Home" => "index.md",
    "Getting Started" => "getting_started.md",
    "Benchmarks" => "benchmarks.md"

]

makedocs(
    sitename="shos-QSimulator.jl",
    pages=enabled_pages
    
)

deploydocs(
    repo = "https://github.com/spherical-tensor/QSimulator.jl",
)
