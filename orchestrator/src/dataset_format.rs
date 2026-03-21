use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use log::info;

/// Supported dataset formats that can be transparently converted
/// to raw packed `f32` binary before training begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetFormat {
    /// Comma-separated values — no header, all columns are `f32`.
    Csv,
    /// Tab-separated values — no header, all columns are `f32`.
    Tsv,
}

impl DatasetFormat {
    /// Infers the dataset format from the file extension.
    ///
    /// # Returns
    /// `Some(DatasetFormat)` for known delimited extensions, `None` for raw binary.
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("csv") => Some(Self::Csv),
            Some("tsv") => Some(Self::Tsv),
            _ => None,
        }
    }

    fn separator(self) -> char {
        match self {
            Self::Csv => ',',
            Self::Tsv => '\t',
        }
    }
}

/// Converts a delimited dataset file (CSV/TSV) to a raw packed `f32` binary file.
///
/// Processes the source file line by line. If the first non-empty row cannot be
/// fully parsed as `f32` values it is treated as a header and skipped automatically.
///
/// The output file is placed next to the source with a `.bin` extension. If it
/// already exists, conversion is skipped and the existing path is returned.
///
/// # Args
/// * `src` - Path to the source delimited file.
/// * `format` - The dataset format to use for parsing.
///
/// # Returns
/// The path to the converted binary file.
///
/// # Errors
/// Returns an `io::Error` if the file cannot be read or written, or if any
/// field in a data row cannot be parsed as `f32`.
pub fn convert_to_binary(src: &Path, format: DatasetFormat) -> io::Result<PathBuf> {
    let out = src.with_extension("bin");

    if out.exists() {
        info!(
            "binary cache already exists at {}, skipping conversion",
            out.display()
        );
        return Ok(out);
    }

    info!("converting {} to binary at {}", src.display(), out.display());

    let sep = format.separator();
    let reader = BufReader::new(File::open(src)?);
    let mut writer = BufWriter::new(File::create(&out)?);

    write_all_rows(reader, sep, &mut writer)?;

    writer.flush()?;

    info!(
        "conversion complete: {} bytes written to {}",
        std::fs::metadata(&out)?.len(),
        out.display()
    );

    Ok(out)
}

fn write_all_rows(
    reader: BufReader<File>,
    sep: char,
    writer: &mut BufWriter<File>,
) -> io::Result<()> {
    let mut lines = reader.lines().enumerate();

    if let Some((n, first)) = find_first_data_line(&mut lines, sep)? {
        write_row(&first, sep, n + 1, writer)?;
    }

    for (line_n, line) in lines {
        let line = line?;
        let trimmed = line.trim();

        if !trimmed.is_empty() {
            write_row(trimmed, sep, line_n + 1, writer)?;
        }
    }

    Ok(())
}

fn find_first_data_line<I>(lines: &mut I, sep: char) -> io::Result<Option<(usize, String)>>
where
    I: Iterator<Item = (usize, io::Result<String>)>,
{
    for (n, line) in lines {
        let line = line?;
        let trimmed = line.trim().to_owned();

        if trimmed.is_empty() {
            continue;
        }

        if !is_data_row(&trimmed, sep) {
            info!("skipping header row: {trimmed:?}");
            return Ok(None);
        }

        return Ok(Some((n, trimmed)));
    }

    Ok(None)
}

fn is_data_row(line: &str, sep: char) -> bool {
    line.split(sep)
        .all(|field| field.trim().parse::<f32>().is_ok())
}

fn write_row(
    line: &str,
    sep: char,
    line_n: usize,
    writer: &mut BufWriter<File>,
) -> io::Result<()> {
    for field in line.split(sep) {
        let val: f32 = field.trim().parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("line {line_n}: cannot parse field as f32: {field:?}"),
            )
        })?;
        writer.write_all(bytemuck::bytes_of(&val))?;
    }
    Ok(())
}