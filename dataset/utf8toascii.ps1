$FolderArray=@("berlin_germany.imposm-geojson", "bogota_colombia.imposm-geojson", "budapest_hungary.imposm-geojson", "cambridge_uk.imposm-geojson", "cape-town_south-africa.imposm-geojson", "chicago_illinois.imposm-geojson", "florence_italy.imposm-geojson", "izmir_turkey.imposm-geojson", "leicester_uk.imposm-geojson", "marseille_france.imposm-geojson", "new-orleans_usa.imposm-geojson", "san-diego_california.imposm-geojson", "tehran_iran.imposm-geojson", "ulaanbaatar_mongolia.imposm-geojson", "valencia_spain.imposm-geojson", "varna_bulgaria.imposm-geojson")
$path="D:\ML PROJE\Mapzen\test\"
ForEach($folder in $FolderArray){
$sourcePath =$path
$sourcePath +=$folder
$destinationPath =$sourcePath
if (!(Test-Path $destinationPath))
{
    New-Item -ItemType Directory -Path $destinationPath
}
Get-ChildItem -Path $sourcePath -File | ForEach-Object {
 Write-Host "Converting $_" 
 $content = Get-Content $_.FullName
 Set-content (Join-Path -Path $destinationPath -ChildPath $_) -Encoding Ascii -Value $content
}
}