// Naive UI theme overrides, keeps built-in components (upload, messages)
// visually consistent with the app's warm "light through skin" tokens.

const fontBody =
  "'Hanken Grotesk Variable', 'Hanken Grotesk', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"

export const themeOverrides = {
  common: {
    fontFamily: fontBody,
    fontWeight: '400',
    fontWeightStrong: '700',

    primaryColor: '#f4726b',
    primaryColorHover: '#f78a84',
    primaryColorPressed: '#d9544d',
    primaryColorSuppl: '#f4726b',

    infoColor: '#f5a623',
    successColor: '#2f9e77',
    errorColor: '#d64550',
    warningColor: '#e08a1e',

    textColorBase: '#2a211d',
    textColor1: '#2a211d',
    textColor2: '#4a3c35',
    textColor3: '#6b5d55',

    bodyColor: '#fbf6f1',
    cardColor: '#ffffff',
    modalColor: '#ffffff',
    popoverColor: '#ffffff',

    borderColor: '#ece1d6',
    borderRadius: '12px',
    borderRadiusSmall: '9px',

    lineHeight: '1.6',
  },
  Button: {
    borderRadiusMedium: '999px',
    borderRadiusLarge: '999px',
    fontWeight: '700',
    heightMedium: '44px',
    heightLarge: '52px',
    paddingMedium: '0 22px',
    paddingLarge: '0 30px',
  },
  Upload: {
    draggerColor: '#fffcf8',
    draggerBorder: '2px dashed #ddcfc1',
    draggerBorderHover: '2px dashed #f4726b',
    borderRadius: '20px',
  },
  Message: {
    borderRadius: '14px',
    padding: '14px 18px',
  },
}
